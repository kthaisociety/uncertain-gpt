/*
=============================================================================
 uncertain/_C.cu — the four arms, driven from Python.

 WHAT THIS FILE IS AND IS NOT

 It is NOT a port. Not one line of the network, the four optimizers, the
 dataloader or the evaluator is reimplemented here. The four .cu files are
 #included verbatim; UNCERTAIN_AS_LIBRARY compiles out their main()s and
 nothing else. Every kernel launched below is the same kernel, in the same
 order, with the same arguments as the standalone binary launches it. If a
 line of arm mathematics ever needs changing, it is changed in the arm's file
 and both consumers move together — the same rule gpt2_kernels.cuh already
 enforces for the network.

 What IS here is the part that used to be main(): the buffers, the loop body,
 the loader, and the lifetime of all of it, lifted into a class so Python can
 own it. That is deliberately the ONLY boundary. See SPEC.md — briefly: the
 forward/backward in gpt2_kernels.cuh is a fused monolith over flat parameter
 and activation slabs, and cutting it into composable autograd ops would cost
 exactly the performance this package exists to ship.

 Evaluation is not reimplemented either. gpt2_kernels.cuh's uq_evaluate takes a
 uq_sample_fn — "give me draw s" — and each arm already supplies one for its
 own `eval` CLI mode. Each Trainer below hands the SAME callback the same way,
 pointed at its live buffers, so a number computed in Python and the same
 number computed by `./train_bnn_gpu eval` come off the same arithmetic.

 ONE TRANSLATION UNIT, ON PURPOSE

 All four arms are compiled together here rather than as four objects. Two
 things fall out of that:

   - gpt2_kernels.cuh's fifteen __global__ kernels have EXTERNAL linkage. Four
     objects each including the header would be four definitions of
     k_attention_forward and the link would fail (or, if they had been made
     static to dodge that, would silently carry four copies of every kernel).
   - the cuBLAS handle is one object, not four. The header now routes it
     through an inline accessor so this holds under any TU layout, but here it
     is true by construction as well.

 Compile time is the price, and it is paid once at install.
=============================================================================
*/

#define UNCERTAIN_AS_LIBRARY 1

#include <torch/extension.h>
#include <pybind11/stl.h>            // std::map / std::vector / std::pair casters

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// The arms, verbatim.
#include "../train_bnn_gpu.cu"       // Bayes-by-Backprop  (Blundell et al. 2015)
#include "../train_bnn_vadam.cu"     // Vadam              (Khan et al. 2018)
#include "../train_ensemble.cu"      // deep ensemble      (Lakshminarayanan 2017)
#include "../train_bnn_lowrank.cu"   // low-rank + diagonal covariance

namespace py = pybind11;

namespace uncertain {

/* ===================================================================== *
 * The cuBLAS handle.
 *
 * gpt2_kernels.cuh gives us exactly one handle per loaded module; what is
 * left is deciding who creates and destroys it. A Trainer cannot own it
 * outright — two Trainers can be alive at once and the second must not be
 * left holding a handle the first destroyed — so it is a refcounted process
 * singleton: created by the first Trainer, destroyed with the last.
 *
 * It is NOT torch's handle (at::cuda::getCurrentCUDABlasHandle). That one
 * follows torch's current stream, while every <<< >>> launch in this codebase
 * goes to the legacy default stream. Borrowing it would put the GEMMs on one
 * stream and the elementwise kernels on another with nothing ordering them —
 * a data race that would show up as occasional wrong numbers, under
 * `with torch.cuda.stream(...)` only. Owning the handle keeps all of it on
 * the default stream, exactly as the binaries have it.
 * ===================================================================== */
static int g_cublas_refs = 0;

static void cublas_acquire() {
    if (g_cublas_refs == 0) CUBLAS_CHECK(cublasCreate(&cublas_handle));
    g_cublas_refs++;
}

static void cublas_release() {
    if (g_cublas_refs > 0 && --g_cublas_refs == 0) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

/* ===================================================================== *
 * Weight init, character for character the loop every arm's main() runs.
 * The caller calls srand() first and this consumes rand() in the same order,
 * so a Trainer's starting weights are the binary's starting weights.
 * ===================================================================== */
static void host_normal_fill(float *h, size_t count, float scale) {
    for (size_t i = 0; i < count; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        h[i] = scale * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
}

/* ===================================================================== *
 * Results handed back to Python. Plain PODs so the compute path never
 * touches a Python object and the GIL can stay released across a step.
 * ===================================================================== */
struct Stats {
    double kl = 0.0, sigma_mean = 0.0, sigma_min = 0.0, sigma_max = 0.0;
};

// UQResult plus the two things the caller had to decide rather than measure:
// how many samples were actually used, and which split they came from. The
// shared harness insists the split is printed with every number and never
// inferred; carrying it on the result is how that survives the trip to Python.
struct EvalOut {
    UQResult r{};
    int  samples = 0;
    bool samples_fixed_by_arm = false;
    std::string split;
};

/* ===================================================================== *
 * TrainerBase — everything the four arms' main()s had in common.
 * ===================================================================== */
class TrainerBase {
public:
    Config   cfg{};
    ParamOff po{};
    ActOff   ao{};
    size_t   n = 0;                 // parameters in the model
    int      BT = 0;
    int      device = 0;

    unsigned long long seed = 20240520ull;
    float lr = 1e-4f, b1 = 0.9f, b2 = 0.999f, aeps = 1e-8f, wd = 0.0f;
    float init_std = 0.02f;
    bool  fixed_batch = false;      // the binaries' `overfit` mode

    int   step_idx = 0;             // completed steps
    float last_ms = 0.0f;

    float *d_acts = nullptr, *d_gacts = nullptr;
    int   *d_inp = nullptr, *d_tgt = nullptr;
    std::vector<int>   h_inp, h_tgt;
    std::vector<float> h_losses;

    Loader ld{nullptr, 0, 0};       // training tokens
    Loader vld{nullptr, 0, 0};      // validation tokens, if a second file was given

    cudaEvent_t ev0{}, ev1{};

    TrainerBase(const Config &c, const std::string &data, const std::string &val_data,
                int device_, unsigned long long seed_, float lr_, float b1_, float b2_,
                float aeps_, float wd_, float init_std_, bool fixed_batch_)
        : cfg(c), device(device_), seed(seed_), lr(lr_), b1(b1_), b2(b2_),
          aeps(aeps_), wd(wd_), init_std(init_std_), fixed_batch(fixed_batch_) {
        try {
            validate(cfg);
            CUDA_CHECK(cudaSetDevice(device));
            cublas_acquire();
            held_cublas_ = true;

            po = param_offsets(cfg);
            ao = act_offsets(cfg);
            n  = po.total;
            BT = cfg.B * cfg.T;

            d_acts  = dmalloc(ao.total * sizeof(float));
            d_gacts = dmalloc(ao.total * sizeof(float));
            d_inp   = (int *)dmalloc((size_t)BT * sizeof(int));
            d_tgt   = (int *)dmalloc((size_t)BT * sizeof(int));

            h_inp.resize(BT);
            h_tgt.resize(BT);
            h_losses.resize(BT);

            loader_init(&ld, data.c_str());
            if (!val_data.empty()) loader_init(&vld, val_data.c_str());

            CUDA_CHECK(cudaEventCreate(&ev0));
            CUDA_CHECK(cudaEventCreate(&ev1));
            events_ = true;
        } catch (...) {
            teardown();
            throw;
        }
    }

    virtual ~TrainerBase() { teardown(); }

    TrainerBase(const TrainerBase &) = delete;
    TrainerBase &operator=(const TrainerBase &) = delete;

    // ---- arm hooks --------------------------------------------------------
    virtual const char *arm() const = 0;
    virtual double step() = 0;
    virtual Stats  stats() = 0;
    virtual void   save(const std::string &tag) = 0;
    virtual std::map<std::string, torch::Tensor> parameters() = 0;
    virtual std::map<std::string, torch::Tensor> buffers() = 0;
    // The arm's uq_sample_fn and its context — bbb_draw/vadam_draw/ens_draw/
    // lowr_draw<R>, straight out of the arm's own file, pointed at this
    // Trainer's live buffers. The extension supplies no sampling code of its
    // own, so a Python eval and a `./train_bnn_gpu eval` draw the same way.
    virtual uq_sample_fn sample_fn() = 0;
    virtual void *sample_ctx() = 0;
    // What the S-sample mixture is called in the report. The control arm keeps
    // saying "ensemble" so its output does not change.
    virtual const char *mix_label() const { return "posterior"; }
    // Load a checkpoint written by save() or by the standalone binary.
    virtual int load(const std::string &tag, int steps) = 0;
    virtual int max_predictors() const { return INT_MAX; }
    virtual int total_steps() const { return -1; }   // -1 = open ended

    // The arm's own settings, reported through the base rather than by binding
    // four derived classes. Binding them would make every factory return a
    // shared_ptr<TrainerBase> that pybind has to downcast into a
    // shared_ptr<Derived> holder — which it does by reinterpret_cast, and which
    // is only benign because this hierarchy happens to be single and
    // non-virtual. Not worth depending on for a handful of read-only numbers.
    virtual std::map<std::string, double> hyperparams() const { return {}; }
    virtual std::vector<int> seeds() const { return {}; }

    // ---- shared machinery -------------------------------------------------
    long train_tokens() const { return ld.n; }

    float *dmalloc(size_t bytes) {
        void *p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, bytes));
        owned_.push_back(p);
        return (float *)p;
    }

    void use_device() { CUDA_CHECK(cudaSetDevice(device)); }

    void next_batch() {
        if (fixed_batch && batch_loaded_) return;   // the binaries' overfit batch
        loader_next(&ld, h_inp.data(), h_tgt.data(), BT);
        upload_batch();
    }

    void upload_batch() {
        CUDA_CHECK(cudaMemcpy(d_inp, h_inp.data(), (size_t)BT * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt.data(), (size_t)BT * sizeof(int),
                              cudaMemcpyHostToDevice));
        batch_loaded_ = true;
    }

    torch::Tensor view(const float *p, int64_t numel) const {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32)
                                          .device(torch::kCUDA, device);
        // No deleter: the tensor is a VIEW of memory this Trainer owns. Its
        // validity ends with the Trainer. Documented in SPEC.md and __init__.py.
        return torch::from_blob(const_cast<float *>(p), {numel}, opts);
    }

    /* ---- evaluation: the shared harness, unchanged ---------------------
       gpt2_kernels.cuh's uq_evaluate does all of it — S draws per position,
       the mixture, the aleatoric/epistemic split, ECE, the OOD sweep and the
       identity gate. The extension contributes nothing but the buffers and the
       arm's own uq_sample_fn, which is the same callback the binary's `eval`
       mode passes. So `t.evaluate()` and `./train_bnn_gpu eval` are the same
       arithmetic on the same draws, and there is one copy of it.

       Held-out data: the validation file if one was given, otherwise the
       training stream from wherever the cursor stands — after k steps that is
       exactly the text training has not reached, which is the choice
       train_ensemble.cu makes with `ld.pos = STEPS * BT`. The cursor and the
       current batch are both restored, so evaluating never perturbs training.
    -------------------------------------------------------------------- */
    EvalOut evaluate(int batches, int samples, const std::string &split) {
        use_device();
        if (batches < 1) throw std::runtime_error("evaluate: need at least one batch");
        int S = samples > 0 ? samples : 5;
        const bool fixed_S = (max_predictors() != INT_MAX);
        if (fixed_S) S = max_predictors();     // the ensemble's S IS its member count
        if (S < 1) throw std::runtime_error("evaluate: need at least one sample");

        Loader *src = vld.tokens ? &vld : &ld;
        long saved_pos = src->pos;

        // Evaluating borrows d_inp/d_tgt and the host batch, and its OOD sweep
        // overwrites them with random ids. Put the training batch back: with
        // fixed_batch=True it is the ONLY batch the run ever sees, and silently
        // swapping it would turn the overfit smoke test into something else.
        bool had_batch = batch_loaded_;
        std::vector<int> save_inp, save_tgt;
        if (had_batch) { save_inp = h_inp; save_tgt = h_tgt; }

        UQSpec spec;
        spec.cfg = cfg;
        spec.S = S;
        spec.nbatches = batches;
        spec.arm = arm();
        spec.split = split.c_str();
        spec.mix_label = mix_label();
        spec.s_fixed_by_arm = fixed_S ? 1 : 0;
        spec.fixed_batch = fixed_batch ? 1 : 0;

        EvalOut out;
        int bad = uq_evaluate(&spec, sample_fn(), sample_ctx(), src, d_acts,
                              d_inp, d_tgt, h_inp.data(), h_tgt.data(),
                              h_losses.data(), &out.r);

        src->pos = saved_pos;
        if (had_batch) { h_inp = save_inp; h_tgt = save_tgt; upload_batch(); }
        else batch_loaded_ = false;

        // The harness's correctness gate, kept as a gate. total = alea + epi is
        // exact algebra for any S and any sampler, so a violation means the
        // PROBABILITIES are wrong — never a modelling problem. The binaries
        // exit nonzero here; raising is the same refusal to report the number.
        if (bad)
            throw std::runtime_error(
                "identity violation: |total-(aleatoric+epistemic)| exceeded 1e-5. "
                "See the note over the harness in gpt2_kernels.cuh — this can only "
                "mean the predictive probabilities are wrong.");

        out.samples = S;
        out.samples_fixed_by_arm = fixed_S;
        out.split = split;
        return out;
    }

    // name -> (offset in the flat parameter vector, shape). Lets Python slice
    // named tensors out of parameters() with no extra C++ and no copy.
    std::map<std::string, std::pair<int64_t, std::vector<int64_t>>> layout() const {
        std::map<std::string, std::pair<int64_t, std::vector<int64_t>>> m;
        const int64_t C = cfg.C, V = cfg.V, maxT = cfg.maxT;
        m["wte"] = {(int64_t)po.wte, {V, C}};
        m["wpe"] = {(int64_t)po.wpe, {maxT, C}};
        for (int l = 0; l < cfg.L; l++) {
            std::string b = "h" + std::to_string(l) + ".";
            m[b + "ln1.weight"]      = {(int64_t)PBLK(po, l, lng),  {C}};
            m[b + "ln1.bias"]        = {(int64_t)PBLK(po, l, lnb),  {C}};
            m[b + "attn.qkv.weight"] = {(int64_t)PBLK(po, l, qw),   {C, 3 * C}};
            m[b + "attn.qkv.bias"]   = {(int64_t)PBLK(po, l, qb),   {3 * C}};
            m[b + "attn.proj.weight"]= {(int64_t)PBLK(po, l, pw),   {C, C}};
            m[b + "attn.proj.bias"]  = {(int64_t)PBLK(po, l, pb),   {C}};
            m[b + "ln2.weight"]      = {(int64_t)PBLK(po, l, ln2g), {C}};
            m[b + "ln2.bias"]        = {(int64_t)PBLK(po, l, ln2b), {C}};
            m[b + "mlp.fc.weight"]   = {(int64_t)PBLK(po, l, w1),   {C, 4 * C}};
            m[b + "mlp.fc.bias"]     = {(int64_t)PBLK(po, l, b1),   {4 * C}};
            m[b + "mlp.proj.weight"] = {(int64_t)PBLK(po, l, w2),   {4 * C, C}};
            m[b + "mlp.proj.bias"]   = {(int64_t)PBLK(po, l, b2),   {C}};
        }
        m["lnf.weight"]  = {(int64_t)po.flng, {C}};
        m["lnf.bias"]    = {(int64_t)po.flnb, {C}};
        m["head.weight"] = {(int64_t)po.hw,   {C, V}};
        m["head.bias"]   = {(int64_t)po.hb,   {V}};
        return m;
    }

protected:
    // Wrap the timed region exactly as the binaries do: an event either side
    // of the whole step, then a synchronize. Same measurement, same sync point.
    void tic() { CUDA_CHECK(cudaEventRecord(ev0)); }
    void toc() {
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&last_ms, ev0, ev1));
    }

    static void validate(const Config &c) {
        if (c.L < 1)  throw std::runtime_error("n_layer must be >= 1");
        if (c.NH < 1) throw std::runtime_error("n_head must be >= 1");
        if (c.C < 1 || c.C % c.NH != 0)
            throw std::runtime_error("n_embd must be a positive multiple of n_head");
        if (c.C / c.NH != HS)
            throw std::runtime_error("n_embd / n_head must equal 64: the FlashAttention "
                                     "kernel holds a head in HS=64 registers and HS is a "
                                     "compile-time constant in gpt2_kernels.cuh");
        if (c.B < 1 || c.T < 1) throw std::runtime_error("batch_size and seq_len must be >= 1");
        if (c.T > c.maxT) throw std::runtime_error("seq_len must be <= max_seq_len");
        if (c.V < 2) throw std::runtime_error("vocab_size must be >= 2");
    }

    bool batch_loaded_ = false;

private:
    void teardown() {
        if (events_) {
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
            events_ = false;
        }
        for (void *p : owned_) cudaFree(p);
        owned_.clear();
        if (ld.tokens)  { free(ld.tokens);  ld.tokens = nullptr; }
        if (vld.tokens) { free(vld.tokens); vld.tokens = nullptr; }
        if (held_cublas_) { cublas_release(); held_cublas_ = false; }
    }

    std::vector<void *> owned_;
    bool events_ = false;
    bool held_cublas_ = false;
};

/* ===================================================================== *
 * Arm: Bayes-by-Backprop.   train_bnn_gpu.cu
 * ===================================================================== */
class BBBTrainer : public TrainerBase {
public:
    float *d_mu, *d_rho, *d_w, *d_dw, *d_dmu, *d_drho;
    float *d_m_mu, *d_v_mu, *d_m_rho, *d_v_rho;
    float *d_klp, *d_sigp, *d_smin, *d_smax;
    std::vector<float> h_klp, h_sigp, h_smin, h_smax;
    float sigma_prior, klw, rho_init;
    int grid_n, stat_blocks = 1024;
    BBBCtx ctx{};

    BBBTrainer(const Config &c, const std::string &data, const std::string &val_data,
               int device_, unsigned long long seed_, float lr_, float b1_, float b2_,
               float aeps_, float wd_, float init_std_, bool fixed_batch_,
               float sigma_prior_, double kl_weight_, float rho_init_)
        : TrainerBase(c, data, val_data, device_, seed_, lr_, b1_, b2_, aeps_, wd_,
                      init_std_, fixed_batch_),
          sigma_prior(sigma_prior_), rho_init(rho_init_) {
        size_t pb = n * sizeof(float);
        d_mu    = dmalloc(pb); d_rho   = dmalloc(pb);
        d_w     = dmalloc(pb); d_dw    = dmalloc(pb);
        d_dmu   = dmalloc(pb); d_drho  = dmalloc(pb);
        d_m_mu  = dmalloc(pb); d_v_mu  = dmalloc(pb);
        d_m_rho = dmalloc(pb); d_v_rho = dmalloc(pb);
        d_klp  = dmalloc(stat_blocks * sizeof(float));
        d_sigp = dmalloc(stat_blocks * sizeof(float));
        d_smin = dmalloc(stat_blocks * sizeof(float));
        d_smax = dmalloc(stat_blocks * sizeof(float));
        h_klp.resize(stat_blocks); h_sigp.resize(stat_blocks);
        h_smin.resize(stat_blocks); h_smax.resize(stat_blocks);

        // The ELBO's KL weight is DERIVED, not tuned: KL is per-dataset, the
        // reported NLL is per-token, so the weight is 1/N. Hardcoding it cost
        // 1.0 nats once — see train_bnn_gpu.cu's header. kl_weight < 0 means
        // "derive it"; 0 disables the prior term (the overfit smoke test).
        klw = (kl_weight_ < 0.0) ? (float)(1.0 / (double)ld.n) : (float)kl_weight_;

        srand(1337);
        std::vector<float> h(n);
        host_normal_fill(h.data(), n, init_std);
        CUDA_CHECK(cudaMemcpy(d_mu, h.data(), pb, cudaMemcpyHostToDevice));
        std::fill(h.begin(), h.end(), rho_init);
        CUDA_CHECK(cudaMemcpy(d_rho, h.data(), pb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_m_mu, 0, pb));  CUDA_CHECK(cudaMemset(d_v_mu, 0, pb));
        CUDA_CHECK(cudaMemset(d_m_rho, 0, pb)); CUDA_CHECK(cudaMemset(d_v_rho, 0, pb));

        grid_n = ceil_div((int)n, BS);
    }

    const char *arm() const override { return "bbb"; }

    std::map<std::string, double> hyperparams() const override {
        return {{"kl_weight", klw}, {"sigma_prior", sigma_prior},
                {"rho_init", rho_init}};
    }

    double step() override {
        use_device();
        size_t pb = n * sizeof(float);
        int s = step_idx + 1;                    // Philox counter, 1-based as in main()
        tic();
        next_batch();

        k_bayes_sample<<<grid_n, BS>>>(d_w, d_mu, d_rho, n, seed, s);

        CUDA_CHECK(cudaMemset(d_dw,    0, pb));
        CUDA_CHECK(cudaMemset(d_dmu,   0, pb));
        CUDA_CHECK(cudaMemset(d_drho,  0, pb));
        CUDA_CHECK(cudaMemset(d_gacts, 0, ao.total * sizeof(float)));

        float nll = gpt2_forward(d_acts, d_w, d_inp, d_tgt, cfg, h_losses.data());
        gpt2_backward(d_dw, d_gacts, d_acts, d_w, d_inp, d_tgt, cfg);

        k_bayes_backward<<<grid_n, BS>>>(d_dmu, d_drho, d_dw, d_rho, n, seed, s);
        if (klw > 0.0f)
            k_kl_backward<<<grid_n, BS>>>(d_dmu, d_drho, d_mu, d_rho, n, sigma_prior, klw);

        float bc1 = 1.0f - powf(b1, (float)s);
        float bc2 = 1.0f - powf(b2, (float)s);
        k_adamw<<<grid_n, BS>>>(d_mu,  d_dmu,  d_m_mu,  d_v_mu,  n, bc1, bc2,
                                lr, b1, b2, aeps, wd);
        k_adamw<<<grid_n, BS>>>(d_rho, d_drho, d_m_rho, d_v_rho, n, bc1, bc2,
                                lr, b1, b2, aeps, wd);
        toc();
        step_idx = s;
        return (double)nll;
    }

    Stats stats() override {
        use_device();
        k_kl_stats<<<stat_blocks, BS>>>(d_klp, d_sigp, d_smin, d_smax,
                                        d_mu, d_rho, n, sigma_prior);
        size_t sb = stat_blocks * sizeof(float);
        CUDA_CHECK(cudaMemcpy(h_klp.data(),  d_klp,  sb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sigp.data(), d_sigp, sb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_smin.data(), d_smin, sb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_smax.data(), d_smax, sb, cudaMemcpyDeviceToHost));
        Stats st;
        double kl = 0.0, sg = 0.0;
        float lo = 1e30f, hi = -1e30f;
        for (int i = 0; i < stat_blocks; i++) {
            kl += h_klp[i]; sg += h_sigp[i];
            if (h_smin[i] < lo) lo = h_smin[i];
            if (h_smax[i] > hi) hi = h_smax[i];
        }
        st.kl = kl; st.sigma_mean = sg / (double)n;
        st.sigma_min = lo; st.sigma_max = hi;
        return st;
    }

    // bbb_draw, from train_bnn_gpu.cu, pointed at these buffers.
    uq_sample_fn sample_fn() override { return bbb_draw; }
    void *sample_ctx() override {
        ctx.w = d_w; ctx.mu = d_mu; ctx.rho = d_rho;
        ctx.n = n;   ctx.grid = grid_n;
        return &ctx;
    }

    void save(const std::string &tag) override {
        use_device();
        ckpt_save(tag.c_str(), "mu",  d_mu,  n, cfg, step_idx);
        ckpt_save(tag.c_str(), "rho", d_rho, n, cfg, step_idx);
    }

    int load(const std::string &tag, int steps) override {
        use_device();
        int a = ckpt_load(tag.c_str(), "mu",  d_mu,  n, cfg, steps);
        int b = ckpt_load(tag.c_str(), "rho", d_rho, n, cfg, steps);
        if (a < 0 || b < 0) throw std::runtime_error("ckpt_load failed; see stdout");
        return a;
    }

    std::map<std::string, torch::Tensor> parameters() override {
        return {{"mu", view(d_mu, n)}, {"rho", view(d_rho, n)}};
    }
    std::map<std::string, torch::Tensor> buffers() override {
        return {{"w", view(d_w, n)},         {"dw", view(d_dw, n)},
                {"dmu", view(d_dmu, n)},     {"drho", view(d_drho, n)},
                {"m_mu", view(d_m_mu, n)},   {"v_mu", view(d_v_mu, n)},
                {"m_rho", view(d_m_rho, n)}, {"v_rho", view(d_v_rho, n)}};
    }
};

/* ===================================================================== *
 * Arm: Vadam.   train_bnn_vadam.cu
 * ===================================================================== */
class VadamTrainer : public TrainerBase {
public:
    float *d_mu, *d_m, *d_s, *d_w, *d_dw;
    float *d_klp, *d_sigp, *d_smin, *d_smax;
    std::vector<float> h_klp, h_sigp, h_smin, h_smax;
    float lam, N, s0;
    int grid_n, stat_blocks = 1024;
    VadamCtx ctx{};

    VadamTrainer(const Config &c, const std::string &data, const std::string &val_data,
                 int device_, unsigned long long seed_, float lr_, float b1_, float b2_,
                 float aeps_, float wd_, float init_std_, bool fixed_batch_,
                 double lam_, float s0_)
        : TrainerBase(c, data, val_data, device_, seed_, lr_, b1_, b2_, aeps_, wd_,
                      init_std_, fixed_batch_),
          s0(s0_) {
        size_t pb = n * sizeof(float);
        d_mu = dmalloc(pb); d_m = dmalloc(pb); d_s = dmalloc(pb);
        d_w  = dmalloc(pb); d_dw = dmalloc(pb);
        d_klp  = dmalloc(stat_blocks * sizeof(float));
        d_sigp = dmalloc(stat_blocks * sizeof(float));
        d_smin = dmalloc(stat_blocks * sizeof(float));
        d_smax = dmalloc(stat_blocks * sizeof(float));
        h_klp.resize(stat_blocks); h_sigp.resize(stat_blocks);
        h_smin.resize(stat_blocks); h_smax.resize(stat_blocks);

        // One data example = one TOKEN, because the loss is a mean over tokens.
        N = (float)ld.n;
        // lambda < 0 means derive it: the prior std should match the scale the
        // weights are initialized at, so lambda = 1/init_std^2. At 0.02 that is
        // 2500 and the step-1 sigma is exactly 0.02. A stale 100 here (sigma_p
        // = 0.1, five times the init scale) is what diverged this arm.
        lam = (lam_ < 0.0) ? 1.0f / (init_std * init_std) : (float)lam_;

        srand(1337);
        std::vector<float> h(n);
        host_normal_fill(h.data(), n, init_std);
        CUDA_CHECK(cudaMemcpy(d_mu, h.data(), pb, cudaMemcpyHostToDevice));
        if (s0 == 0.0f) {
            CUDA_CHECK(cudaMemset(d_s, 0, pb));
        } else {
            std::fill(h.begin(), h.end(), s0);
            CUDA_CHECK(cudaMemcpy(d_s, h.data(), pb, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemset(d_m, 0, pb));

        grid_n = ceil_div((int)n, BS);
    }

    const char *arm() const override { return "vadam"; }

    std::map<std::string, double> hyperparams() const override {
        return {{"lam", lam}, {"N", N}, {"s0", s0}};
    }

    double step() override {
        use_device();
        size_t pb = n * sizeof(float);
        int s = step_idx + 1;
        tic();
        next_batch();                                               // line 4

        k_vadam_sample<<<grid_n, BS>>>(d_w, d_mu, d_s, n, N, lam, seed, s);  // 1-3

        CUDA_CHECK(cudaMemset(d_dw,    0, pb));
        CUDA_CHECK(cudaMemset(d_gacts, 0, ao.total * sizeof(float)));

        float nll = gpt2_forward(d_acts, d_w, d_inp, d_tgt, cfg, h_losses.data());
        gpt2_backward(d_dw, d_gacts, d_acts, d_w, d_inp, d_tgt, cfg);        // line 5

        float bc1 = 1.0f - powf(b1, (float)s);
        float bc2 = 1.0f - powf(b2, (float)s);
        k_vadam_update<<<grid_n, BS>>>(d_mu, d_m, d_s, d_dw, n, N, lam,
                                       bc1, bc2, lr, b1, b2);                // 6-9
        toc();
        step_idx = s;
        return (double)nll;
    }

    Stats stats() override {
        use_device();
        k_vadam_stats<<<stat_blocks, BS>>>(d_klp, d_sigp, d_smin, d_smax,
                                           d_mu, d_s, n, N, lam);
        size_t sb = stat_blocks * sizeof(float);
        CUDA_CHECK(cudaMemcpy(h_klp.data(),  d_klp,  sb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sigp.data(), d_sigp, sb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_smin.data(), d_smin, sb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_smax.data(), d_smax, sb, cudaMemcpyDeviceToHost));
        Stats st;
        double kl = 0.0, sg = 0.0;
        float lo = 1e30f, hi = -1e30f;
        for (int i = 0; i < stat_blocks; i++) {
            kl += h_klp[i]; sg += h_sigp[i];
            if (h_smin[i] < lo) lo = h_smin[i];
            if (h_smax[i] > hi) hi = h_smax[i];
        }
        st.kl = kl; st.sigma_mean = sg / (double)n;
        st.sigma_min = lo; st.sigma_max = hi;
        return st;
    }

    // vadam_draw, from train_bnn_vadam.cu. N and lambda travel in the context
    // because sigma = 1/sqrt(N*s + lambda) is DERIVED, so a draw needs both.
    uq_sample_fn sample_fn() override { return vadam_draw; }
    void *sample_ctx() override {
        ctx.w = d_w; ctx.mu = d_mu; ctx.s = d_s;
        ctx.n = n;   ctx.grid = grid_n; ctx.N = N; ctx.lambda = lam;
        return &ctx;
    }

    void save(const std::string &tag) override {
        use_device();
        ckpt_save(tag.c_str(), "mu", d_mu, n, cfg, step_idx);
        ckpt_save(tag.c_str(), "s",  d_s,  n, cfg, step_idx);   // sigma derives from s
    }

    int load(const std::string &tag, int steps) override {
        use_device();
        int a = ckpt_load(tag.c_str(), "mu", d_mu, n, cfg, steps);
        int b = ckpt_load(tag.c_str(), "s",  d_s,  n, cfg, steps);
        if (a < 0 || b < 0) throw std::runtime_error("ckpt_load failed; see stdout");
        return a;
    }

    std::map<std::string, torch::Tensor> parameters() override {
        return {{"mu", view(d_mu, n)}, {"s", view(d_s, n)}};
    }
    std::map<std::string, torch::Tensor> buffers() override {
        return {{"m", view(d_m, n)}, {"w", view(d_w, n)}, {"dw", view(d_dw, n)}};
    }
};

/* ===================================================================== *
 * Arm: deep ensemble (the control).   train_ensemble.cu
 *
 * The one place an arm's structure shows through the API. The binary trains
 * member 0 to completion, then member 1, and so on, reusing ONE pair of AdamW
 * moment buffers. Interleaving the members instead would need E pairs, and
 * this arm's memory footprint is a reported result — so step() walks the
 * members in the binary's order and `steps_per_member` is what says where one
 * ends and the next begins.
 * ===================================================================== */
class EnsembleTrainer : public TrainerBase {
public:
    std::vector<float *> d_member;
    float *d_m, *d_v, *d_dw;
    std::vector<int> member_seeds;
    int steps_per_member;
    EnsCtx ctx{};
    int grid_n;

    EnsembleTrainer(const Config &c, const std::string &data, const std::string &val_data,
                    int device_, unsigned long long seed_, float lr_, float b1_, float b2_,
                    float aeps_, float wd_, float init_std_, bool fixed_batch_,
                    std::vector<int> seeds_, int steps_per_member_)
        : TrainerBase(c, data, val_data, device_, seed_, lr_, b1_, b2_, aeps_, wd_,
                      init_std_, fixed_batch_),
          member_seeds(std::move(seeds_)), steps_per_member(steps_per_member_) {
        if (member_seeds.empty()) throw std::runtime_error("ensemble needs at least one seed");
        // EnsCtx in train_ensemble.cu holds `const float *member[E_MEMBERS]`,
        // and ens_draw indexes it directly. Rather than keep a second, wider
        // context here — which would be a copy of the arm's code — the member
        // count is capped at the arm's own E_MEMBERS. That is the count the
        // published comparison uses.
        if ((int)member_seeds.size() > E_MEMBERS)
            throw std::runtime_error(
                "this build supports at most " + std::to_string(E_MEMBERS) +
                " ensemble members (E_MEMBERS in train_ensemble.cu)");
        if (steps_per_member < 1) throw std::runtime_error("steps_per_member must be >= 1");
        size_t pb = n * sizeof(float);
        d_member.resize(member_seeds.size());
        for (size_t e = 0; e < member_seeds.size(); e++) d_member[e] = dmalloc(pb);
        d_m  = dmalloc(pb);
        d_v  = dmalloc(pb);
        d_dw = dmalloc(pb);

        // Every member up front. The binary initializes member e just before
        // training it, but each init calls srand(SEEDS[e]) first and nothing
        // else draws rand() in between, so the streams are independent and
        // doing them all here produces the same weights.
        std::vector<float> h(n);
        for (size_t e = 0; e < member_seeds.size(); e++) {
            srand(member_seeds[e]);
            host_normal_fill(h.data(), n, init_std);
            CUDA_CHECK(cudaMemcpy(d_member[e], h.data(), pb, cudaMemcpyHostToDevice));
        }
        grid_n = ceil_div((int)n, BS);
    }

    const char *arm() const override { return "ens"; }
    int max_predictors() const override { return (int)member_seeds.size(); }
    int total_steps() const override { return (int)member_seeds.size() * steps_per_member; }

    std::map<std::string, double> hyperparams() const override {
        return {{"members", (double)member_seeds.size()},
                {"steps_per_member", (double)steps_per_member},
                {"current_member", (double)(step_idx / steps_per_member)}};
    }
    std::vector<int> seeds() const override { return member_seeds; }

    double step() override {
        use_device();
        int e = step_idx / steps_per_member;
        if (e >= (int)member_seeds.size())
            throw std::runtime_error("ensemble: every member has had its "
                                     "steps_per_member steps; training is finished");
        int local = step_idx % steps_per_member;    // 0-based within this member
        size_t pb = n * sizeof(float);

        tic();
        if (local == 0) {                           // a fresh member starts here
            CUDA_CHECK(cudaMemset(d_m, 0, pb));
            CUDA_CHECK(cudaMemset(d_v, 0, pb));
            ld.pos = 0;                             // identical data order per member
        }
        next_batch();
        CUDA_CHECK(cudaMemset(d_dw,    0, pb));
        CUDA_CHECK(cudaMemset(d_gacts, 0, ao.total * sizeof(float)));

        float nll = gpt2_forward(d_acts, d_member[e], d_inp, d_tgt, cfg, h_losses.data());
        gpt2_backward(d_dw, d_gacts, d_acts, d_member[e], d_inp, d_tgt, cfg);

        // bias correction counts THIS member's steps, as in the binary
        float bc1 = 1.0f - powf(b1, (float)(local + 1));
        float bc2 = 1.0f - powf(b2, (float)(local + 1));
        k_adamw<<<grid_n, BS>>>(d_member[e], d_dw, d_m, d_v, n, bc1, bc2,
                                lr, b1, b2, aeps, wd);
        toc();
        step_idx++;
        return (double)nll;
    }

    // Nothing is sampled in this arm; the binary prints kl 0 and sigma 0 too.
    Stats stats() override { return Stats{}; }

    // ens_draw, from train_ensemble.cu: sample e IS member e, no copy.
    uq_sample_fn sample_fn() override { return ens_draw; }
    void *sample_ctx() override {
        for (size_t e = 0; e < member_seeds.size(); e++) ctx.member[e] = d_member[e];
        return &ctx;
    }
    const char *mix_label() const override { return "ensemble"; }

    int load(const std::string &tag, int steps) override {
        use_device();
        int got = -1;
        for (size_t e = 0; e < member_seeds.size(); e++) {
            char w[32];
            snprintf(w, sizeof w, "member%zu", e);
            got = ckpt_load(tag.c_str(), w, d_member[e], n, cfg, steps);
            if (got < 0) throw std::runtime_error("ckpt_load failed; see stdout");
        }
        return got;
    }

    void save(const std::string &tag) override {
        use_device();
        for (size_t e = 0; e < member_seeds.size(); e++) {
            char w[32];
            snprintf(w, sizeof w, "member%zu", e);
            ckpt_save(tag.c_str(), w, d_member[e], n, cfg, step_idx);
        }
    }

    std::map<std::string, torch::Tensor> parameters() override {
        std::map<std::string, torch::Tensor> m;
        for (size_t e = 0; e < member_seeds.size(); e++)
            m["member" + std::to_string(e)] = view(d_member[e], n);
        return m;
    }
    std::map<std::string, torch::Tensor> buffers() override {
        return {{"m", view(d_m, n)}, {"v", view(d_v, n)}, {"dw", view(d_dw, n)}};
    }
};

/* ===================================================================== *
 * Arm: low-rank + diagonal covariance.   train_bnn_lowrank.cu
 *
 * Templated on the rank for the same reason the kernels are: the r-loops must
 * stay compile-time. The factory below instantiates a fixed set of ranks and
 * picks one at run time, which is what turns `RANK` from a build flag into a
 * constructor argument without giving up a single unrolled iteration.
 * ===================================================================== */
template <int R>
class LowRankTrainer : public TrainerBase {
public:
    float *d_theta, *d_w, *d_dw, *d_m, *d_v;
    float *d_A, *d_dA, *d_mA, *d_vA;
    float *d_rhod, *d_drhod, *d_md, *d_vd;
    float *d_z, *d_M, *d_Minv;
    float *dp[7];
    std::vector<float> hp[7];
    std::vector<float> h_M, h_Minv;
    std::vector<double> logdetM;
    std::vector<BLayer> bl;
    int NBL = 0;
    size_t Pb = 0;
    float lam, klw, lr_var, a_init, dvec_init;
    unsigned long long zseed;
    int gA, gB, gN;
    LowRCtx ctx{};
    static constexpr int SB = 256;

    LowRankTrainer(const Config &c, const std::string &data, const std::string &val_data,
                   int device_, unsigned long long seed_, float lr_, float b1_, float b2_,
                   float aeps_, float wd_, float init_std_, bool fixed_batch_,
                   float lam_, double kl_weight_, float lr_var_,
                   float a_init_, float dvec_init_)
        : TrainerBase(c, data, val_data, device_, seed_, lr_, b1_, b2_, aeps_, wd_,
                      init_std_, fixed_batch_),
          lam(lam_), lr_var(lr_var_), a_init(a_init_), dvec_init(dvec_init_) {
        zseed = seed ^ 0x9E3779B9ull;

        // Which tensors are Bayesian, from the arm's own list — not a second
        // copy of it here. lowr_build_layers is the single definition that
        // main(), lowr_eval() and this class all go through.
        NBL = 4 * cfg.L;
        bl.resize(NBL);
        Pb = lowr_build_layers<R>(bl.data(), cfg);

        size_t pb = n * sizeof(float);
        size_t ab = Pb * R * sizeof(float);
        size_t db = Pb * sizeof(float);
        d_theta = dmalloc(pb); d_w  = dmalloc(pb); d_dw = dmalloc(pb);
        d_m     = dmalloc(pb); d_v  = dmalloc(pb);
        d_A     = dmalloc(ab); d_dA = dmalloc(ab);
        d_mA    = dmalloc(ab); d_vA = dmalloc(ab);
        d_rhod  = dmalloc(db); d_drhod = dmalloc(db);
        d_md    = dmalloc(db); d_vd    = dmalloc(db);
        d_z     = dmalloc((size_t)NBL * R * sizeof(float));
        d_M     = dmalloc((size_t)NBL * R * R * sizeof(float));
        d_Minv  = dmalloc((size_t)NBL * R * R * sizeof(float));
        for (int i = 0; i < 7; i++) {
            dp[i] = dmalloc(SB * sizeof(float));
            hp[i].resize(SB);
        }
        h_M.resize((size_t)NBL * R * R);
        h_Minv.resize((size_t)NBL * R * R);
        // zeroed, not merely allocated: with kl_weight = 0 nothing ever writes
        // logdetM and stats() would otherwise read uninitialized memory.
        logdetM.assign(NBL, 0.0);

        klw = (kl_weight_ < 0.0) ? (float)(1.0 / (double)ld.n) : (float)kl_weight_;

        // init: theta, then A, then rhod — one rand() stream, same order as the
        // binary, so the starting point is the binary's starting point.
        srand(1337);
        std::vector<float> h(std::max(n, Pb * R));
        host_normal_fill(h.data(), n, init_std);
        CUDA_CHECK(cudaMemcpy(d_theta, h.data(), pb, cudaMemcpyHostToDevice));
        host_normal_fill(h.data(), Pb * R, a_init);
        CUDA_CHECK(cudaMemcpy(d_A, h.data(), ab, cudaMemcpyHostToDevice));
        const float rhod0 = logf(expf(dvec_init) - 1.0f);   // softplus(rhod) = dvec_init
        std::fill(h.begin(), h.begin() + (std::ptrdiff_t)Pb, rhod0);
        CUDA_CHECK(cudaMemcpy(d_rhod, h.data(), db, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_m,  0, pb)); CUDA_CHECK(cudaMemset(d_v,  0, pb));
        CUDA_CHECK(cudaMemset(d_mA, 0, ab)); CUDA_CHECK(cudaMemset(d_vA, 0, ab));
        CUDA_CHECK(cudaMemset(d_md, 0, db)); CUDA_CHECK(cudaMemset(d_vd, 0, db));

        gA = ceil_div((int)(Pb * R), BS);
        gB = ceil_div((int)Pb, BS);
        gN = ceil_div((int)n, BS);
    }

    const char *arm() const override { return "lowr"; }

    std::map<std::string, double> hyperparams() const override {
        return {{"rank", (double)R}, {"lam", lam}, {"kl_weight", klw},
                {"lr_var", lr_var}, {"a_init", a_init}, {"dvec_init", dvec_init},
                {"bayesian_params", (double)Pb},
                {"bayesian_layers", (double)NBL}};
    }

    double step() override {
        use_device();
        size_t pb = n * sizeof(float);
        size_t ab = Pb * R * sizeof(float);
        size_t db = Pb * sizeof(float);
        int s = step_idx + 1;
        tic();
        next_batch();

        // M = I + A^T D^-1 A, per layer, BEFORE dA is used as a gradient.
        if (klw > 0.0f) {
            const float one = 1.0f, zero = 0.0f;
            for (int l = 0; l < NBL; l++) {
                k_scale_A_dinv<R><<<ceil_div(bl[l].P, BS), BS>>>(
                    d_dA, d_A, d_rhod, bl[l].aoff, bl[l].boff, bl[l].P);
                CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    R, R, bl[l].P, &one, d_dA + bl[l].aoff, R,
                    d_A + bl[l].aoff, R, &zero, d_M + l * R * R, R));
            }
            CUDA_CHECK(cudaMemcpy(h_M.data(), d_M, (size_t)NBL * R * R * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            for (int l = 0; l < NBL; l++) {
                for (int j = 0; j < R; j++) h_M[l * R * R + j * R + j] += 1.0f;
                if (!chol_logdet_inv<R>(h_M.data() + l * R * R,
                                        h_Minv.data() + l * R * R, &logdetM[l]))
                    throw std::runtime_error(std::string("M not positive definite in layer ")
                                             + bl[l].name);
            }
            CUDA_CHECK(cudaMemcpy(d_Minv, h_Minv.data(), (size_t)NBL * R * R * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        k_draw_z<R><<<ceil_div(NBL * R, 128), 128>>>(d_z, NBL, zseed, s);

        CUDA_CHECK(cudaMemcpy(d_w, d_theta, pb, cudaMemcpyDeviceToDevice));
        for (int l = 0; l < NBL; l++)
            k_lowrank_sample<R><<<ceil_div(bl[l].P, BS), BS>>>(d_w, d_theta, d_A, d_rhod,
                d_z + l * R, bl[l].poff, bl[l].aoff, bl[l].boff, bl[l].P, seed, s);

        CUDA_CHECK(cudaMemset(d_dw,    0, pb));
        CUDA_CHECK(cudaMemset(d_dA,    0, ab));
        CUDA_CHECK(cudaMemset(d_drhod, 0, db));
        CUDA_CHECK(cudaMemset(d_gacts, 0, ao.total * sizeof(float)));

        float nll = gpt2_forward(d_acts, d_w, d_inp, d_tgt, cfg, h_losses.data());
        gpt2_backward(d_dw, d_gacts, d_acts, d_w, d_inp, d_tgt, cfg);

        for (int l = 0; l < NBL; l++)
            k_lowrank_backward<R><<<ceil_div(bl[l].P, BS), BS>>>(d_dA, d_drhod, d_dw,
                d_rhod, d_z + l * R, bl[l].poff, bl[l].aoff, bl[l].boff, bl[l].P, seed, s);

        if (klw > 0.0f)
            for (int l = 0; l < NBL; l++)
                k_lowrank_kl_grad<R><<<ceil_div(bl[l].P, BS), BS>>>(d_dw, d_dA, d_drhod,
                    d_theta, d_A, d_rhod, d_Minv + l * R * R,
                    bl[l].poff, bl[l].aoff, bl[l].boff, bl[l].P, lam, klw);

        float bc1 = 1.0f - powf(b1, (float)s), bc2 = 1.0f - powf(b2, (float)s);
        k_adamw<<<gN, BS>>>(d_theta, d_dw,    d_m,  d_v,  n,     bc1, bc2, lr,
                            b1, b2, aeps, wd);
        k_adamw<<<gA, BS>>>(d_A,     d_dA,    d_mA, d_vA, Pb * R, bc1, bc2, lr_var,
                            b1, b2, aeps, wd);
        k_adamw<<<gB, BS>>>(d_rhod,  d_drhod, d_md, d_vd, Pb,     bc1, bc2, lr_var,
                            b1, b2, aeps, wd);
        toc();
        step_idx = s;
        return (double)nll;
    }

    Stats stats() override {
        use_device();
        double a2 = 0, d2 = 0, logd = 0, mu2 = 0, smean = 0;
        float lo = 1e30f, hi = -1e30f;
        for (int l = 0; l < NBL; l++) {
            k_lowrank_stats<R><<<SB, BS>>>(dp[0], dp[1], dp[2], dp[3], dp[4], dp[5], dp[6],
                d_theta, d_A, d_rhod, bl[l].poff, bl[l].aoff, bl[l].boff, bl[l].P);
            for (int q = 0; q < 7; q++)
                CUDA_CHECK(cudaMemcpy(hp[q].data(), dp[q], SB * sizeof(float),
                                      cudaMemcpyDeviceToHost));
            for (int i = 0; i < SB; i++) {
                a2 += hp[0][i]; d2 += hp[1][i]; logd += hp[2][i]; mu2 += hp[3][i];
                if (hp[4][i] < lo) lo = hp[4][i];
                if (hp[5][i] > hi) hi = hp[5][i];
                smean += hp[6][i];
            }
        }
        // KL = 0.5[lambda(||mu||^2+trSigma) - P - logdet Sigma + P log lambda]
        double ldet = logd;
        for (int l = 0; l < NBL; l++) ldet += logdetM[l];
        Stats st;
        st.kl = 0.5 * (lam * (mu2 + a2 + d2) - (double)Pb - ldet
                       + (double)Pb * log((double)lam));
        st.sigma_mean = smean / (double)Pb;
        st.sigma_min = lo; st.sigma_max = hi;
        return st;
    }

    // lowr_draw<R>, from train_bnn_lowrank.cu.
    uq_sample_fn sample_fn() override { return lowr_draw<R>; }
    void *sample_ctx() override {
        ctx.w = d_w; ctx.theta = d_theta; ctx.A = d_A; ctx.rhod = d_rhod;
        ctx.z = d_z; ctx.L = bl.data(); ctx.nbl = NBL; ctx.n = n;
        return &ctx;
    }

    void save(const std::string &tag) override {
        use_device();
        ckpt_save(tag.c_str(), "theta", d_theta, n,      cfg, step_idx);
        ckpt_save(tag.c_str(), "A",     d_A,     Pb * R, cfg, step_idx);
        ckpt_save(tag.c_str(), "rhod",  d_rhod,  Pb,     cfg, step_idx);
    }

    // ckpt_load checks the element count, so loading an r=8 A into an r=4 run
    // fails loudly instead of reading 4 MB of the wrong thing.
    int load(const std::string &tag, int steps) override {
        use_device();
        int a = ckpt_load(tag.c_str(), "theta", d_theta, n,      cfg, steps);
        int b = ckpt_load(tag.c_str(), "A",     d_A,     Pb * R, cfg, steps);
        int c = ckpt_load(tag.c_str(), "rhod",  d_rhod,  Pb,     cfg, steps);
        if (a < 0 || b < 0 || c < 0) throw std::runtime_error("ckpt_load failed; see stdout");
        return a;
    }

    std::map<std::string, torch::Tensor> parameters() override {
        return {{"theta", view(d_theta, n)},
                {"A",     view(d_A, (int64_t)(Pb * R))},
                {"rhod",  view(d_rhod, (int64_t)Pb)}};
    }
    std::map<std::string, torch::Tensor> buffers() override {
        return {{"w", view(d_w, n)},   {"dw", view(d_dw, n)},
                {"m", view(d_m, n)},   {"v",  view(d_v, n)},
                {"mA", view(d_mA, (int64_t)(Pb * R))},
                {"vA", view(d_vA, (int64_t)(Pb * R))},
                {"md", view(d_md, (int64_t)Pb)},
                {"vd", view(d_vd, (int64_t)Pb)},
                {"z",  view(d_z, (int64_t)NBL * R)}};
    }
};

// The ranks this module is built with. Each is a separate instantiation of six
// small elementwise kernels, so the list is cheap; adding one costs a rebuild.
#define UNCERTAIN_LOWRANK_RANKS(X) X(1) X(2) X(4) X(8) X(16) X(32)

static std::vector<int> supported_ranks() {
    std::vector<int> v;
#define COLLECT(R) v.push_back(R);
    UNCERTAIN_LOWRANK_RANKS(COLLECT)
#undef COLLECT
    return v;
}

/* ===================================================================== *
 * Factories + bindings
 * ===================================================================== */

static Config make_config(int B, int T, int C, int V, int NH, int maxT, int L) {
    Config c;
    c.B = B; c.T = T; c.C = C; c.V = V; c.NH = NH; c.maxT = maxT; c.L = L;
    return c;
}

}  // namespace uncertain

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace uncertain;
    m.doc() = "uncertain: the C/CUDA Bayesian GPT-2 trainer, driven from Python";

    py::class_<Stats>(m, "Stats")
        .def_readonly("kl", &Stats::kl)
        .def_readonly("sigma_mean", &Stats::sigma_mean)
        .def_readonly("sigma_min", &Stats::sigma_min)
        .def_readonly("sigma_max", &Stats::sigma_max);

    py::class_<EvalOut>(m, "EvalOut")
        .def_property_readonly("nlpd",         [](const EvalOut &e) { return e.r.nlpd_mix; })
        .def_property_readonly("nlpd_single",  [](const EvalOut &e) { return e.r.nlpd_single; })
        .def_property_readonly("ece",          [](const EvalOut &e) { return e.r.ece; })
        .def_property_readonly("aleatoric",    [](const EvalOut &e) { return e.r.alea; })
        .def_property_readonly("epistemic",    [](const EvalOut &e) { return e.r.epi; })
        .def_property_readonly("epistemic_ood",[](const EvalOut &e) { return e.r.epi_ood; })
        .def_property_readonly("total",        [](const EvalOut &e) { return e.r.total; })
        .def_property_readonly("identity_err", [](const EvalOut &e) { return e.r.id_err_max; })
        .def_property_readonly("positions",    [](const EvalOut &e) { return e.r.npos; })
        .def_property_readonly("positions_ood",[](const EvalOut &e) { return e.r.npos_ood; })
        .def_readonly("samples", &EvalOut::samples)
        .def_readonly("samples_fixed_by_arm", &EvalOut::samples_fixed_by_arm)
        .def_readonly("split", &EvalOut::split);

    py::class_<TrainerBase, std::shared_ptr<TrainerBase>>(m, "Trainer")
        .def("step", &TrainerBase::step, py::call_guard<py::gil_scoped_release>())
        .def("stats", &TrainerBase::stats, py::call_guard<py::gil_scoped_release>())
        .def("evaluate", &TrainerBase::evaluate,
             py::arg("batches"), py::arg("samples"), py::arg("split"),
             py::call_guard<py::gil_scoped_release>())
        .def("save", &TrainerBase::save, py::arg("tag"))
        .def("load", &TrainerBase::load, py::arg("tag"), py::arg("steps"),
             py::call_guard<py::gil_scoped_release>())
        .def("parameters", &TrainerBase::parameters)
        .def("buffers", &TrainerBase::buffers)
        .def("layout", &TrainerBase::layout)
        .def_property_readonly("arm", [](const TrainerBase &t) { return std::string(t.arm()); })
        .def_property_readonly("n_params", [](const TrainerBase &t) { return (int64_t)t.n; })
        .def_property_readonly("n_act", [](const TrainerBase &t) { return (int64_t)t.ao.total; })
        .def_property_readonly("train_tokens", &TrainerBase::train_tokens)
        .def_property_readonly("device", [](const TrainerBase &t) { return t.device; })
        .def_property_readonly("max_predictors", &TrainerBase::max_predictors)
        .def_property_readonly("total_steps", &TrainerBase::total_steps)
        .def_property_readonly("hyperparams", &TrainerBase::hyperparams)
        .def_property_readonly("seeds", &TrainerBase::seeds)
        .def_readonly("step_idx", &TrainerBase::step_idx)
        .def_readonly("last_step_ms", &TrainerBase::last_ms)
        .def_property_readonly("config", [](const TrainerBase &t) {
            py::dict d;
            d["batch_size"] = t.cfg.B;   d["seq_len"] = t.cfg.T;
            d["n_embd"]     = t.cfg.C;   d["vocab_size"] = t.cfg.V;
            d["n_head"]     = t.cfg.NH;  d["max_seq_len"] = t.cfg.maxT;
            d["n_layer"]    = t.cfg.L;
            return d;
        });

    // Arm-specific settings come back through Trainer.hyperparams / .seeds
    // above; the concrete classes are deliberately NOT registered. See the note
    // over TrainerBase::hyperparams.

    /* --- factories ---------------------------------------------------------
       Every argument is NAMED and the Python layer passes them by keyword. The
       common prefix is 18 arguments long and four factories share it; passing
       that positionally is how a `beta2` ends up in `eps` with no error and a
       plausible-looking loss curve. Keywords make that a TypeError instead. */
#define UNCERTAIN_COMMON_ARGS                                                    \
    py::arg("data"), py::arg("val_data"), py::arg("B"), py::arg("T"),            \
    py::arg("C"), py::arg("V"), py::arg("NH"), py::arg("maxT"), py::arg("L"),    \
    py::arg("device"), py::arg("seed"), py::arg("lr"), py::arg("beta1"),         \
    py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"),                   \
    py::arg("init_std"), py::arg("fixed_batch")

    m.def("make_bbb", [](const std::string &data, const std::string &val_data,
                         int B, int T, int C, int V, int NH, int maxT, int L,
                         int device, unsigned long long seed, float lr, float beta1,
                         float beta2, float eps, float weight_decay, float init_std,
                         bool fixed_batch, float sigma_prior, double kl_weight,
                         float rho_init) {
        return std::static_pointer_cast<TrainerBase>(std::make_shared<BBBTrainer>(
            make_config(B, T, C, V, NH, maxT, L), data, val_data, device, seed, lr,
            beta1, beta2, eps, weight_decay, init_std, fixed_batch,
            sigma_prior, kl_weight, rho_init));
    }, UNCERTAIN_COMMON_ARGS, py::arg("sigma_prior"), py::arg("kl_weight"),
       py::arg("rho_init"));

    m.def("make_vadam", [](const std::string &data, const std::string &val_data,
                           int B, int T, int C, int V, int NH, int maxT, int L,
                           int device, unsigned long long seed, float lr, float beta1,
                           float beta2, float eps, float weight_decay, float init_std,
                           bool fixed_batch, double lam, float s0) {
        return std::static_pointer_cast<TrainerBase>(std::make_shared<VadamTrainer>(
            make_config(B, T, C, V, NH, maxT, L), data, val_data, device, seed, lr,
            beta1, beta2, eps, weight_decay, init_std, fixed_batch, lam, s0));
    }, UNCERTAIN_COMMON_ARGS, py::arg("lam"), py::arg("s0"));

    m.def("make_ensemble", [](const std::string &data, const std::string &val_data,
                              int B, int T, int C, int V, int NH, int maxT, int L,
                              int device, unsigned long long seed, float lr, float beta1,
                              float beta2, float eps, float weight_decay, float init_std,
                              bool fixed_batch, std::vector<int> seeds,
                              int steps_per_member) {
        return std::static_pointer_cast<TrainerBase>(std::make_shared<EnsembleTrainer>(
            make_config(B, T, C, V, NH, maxT, L), data, val_data, device, seed, lr,
            beta1, beta2, eps, weight_decay, init_std, fixed_batch,
            std::move(seeds), steps_per_member));
    }, UNCERTAIN_COMMON_ARGS, py::arg("seeds"), py::arg("steps_per_member"));

    m.def("make_lowrank", [](const std::string &data, const std::string &val_data,
                             int B, int T, int C, int V, int NH, int maxT, int L,
                             int device, unsigned long long seed, float lr, float beta1,
                             float beta2, float eps, float weight_decay, float init_std,
                             bool fixed_batch, int rank, float lam, double kl_weight,
                             float lr_var, float a_init, float dvec_init) {
        Config c = make_config(B, T, C, V, NH, maxT, L);
        switch (rank) {
#define CASE(R)                                                                       \
        case R:                                                                       \
            return std::static_pointer_cast<TrainerBase>(                             \
                std::make_shared<LowRankTrainer<R>>(c, data, val_data, device, seed,  \
                    lr, beta1, beta2, eps, weight_decay, init_std, fixed_batch,       \
                    lam, kl_weight, lr_var, a_init, dvec_init));
        UNCERTAIN_LOWRANK_RANKS(CASE)
#undef CASE
        default:
            throw std::invalid_argument(
                "rank must be one of the ranks this module was built with; the "
                "r-loop is unrolled at compile time, so a new rank needs a rebuild "
                "(edit UNCERTAIN_LOWRANK_RANKS in uncertain/_C.cu)");
        }
    }, UNCERTAIN_COMMON_ARGS, py::arg("rank"), py::arg("lam"), py::arg("kl_weight"),
       py::arg("lr_var"), py::arg("a_init"), py::arg("dvec_init"));
#undef UNCERTAIN_COMMON_ARGS

    // The decomposition's own self-test: three cases whose answers are known
    // on paper, pure host arithmetic. No GPU, no data, no model — so this is
    // the cheapest possible proof that the module built and links correctly.
    m.def("selftest", &uq_selftest);

    m.def("supported_ranks", &supported_ranks);
    m.def("default_ensemble_seeds", []() {
        return std::vector<int>(SEEDS, SEEDS + E_MEMBERS);
    });
    m.attr("HEAD_SIZE")     = HS;      // n_embd / n_head must equal this
    m.attr("BLOCK_SIZE")    = BS;
    m.attr("DEFAULT_VOCAB") = VOCAB;
    m.attr("MAX_ENSEMBLE_MEMBERS") = E_MEMBERS;
    m.attr("UQ_BINS")       = UQ_BINS;
    m.attr("DATA_TRAIN")    = DATA_TRAIN;
    m.attr("DATA_VAL")      = DATA_VAL;
    m.attr("DATA_TEST")     = DATA_TEST;
}
