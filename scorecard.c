/*
=============================================================================
 scorecard.c — run every trainer variant and print ONE comparison table.

     make test

 builds everything, runs the four configurations below, parses their logs,
 and prints the scorecard. This exists because the project is now BRANCHING:
 multiple Bayesian methods over one shared network, and "read two logs side
 by side" stops being a comparison the moment there are three.

 The gates are deliberately different per mode:

   overfit  PASS if final NLL < 0.05 — can the machinery memorize one batch?
            This is the correctness gate. (Known: Vadam fails it at
            lambda=100 because sigma has no off switch — its noise floor is
            the prior. That is a finding, not a bug; the scorecard reports
            what is.)
   train    PASS if the loss went down — is it learning on real data?

 Parsing contract: every trainer prints step lines in the SAME format
     step %4d | nll %8.4f | kl %10.1f | sigma %.6f [%.6f, %.6f] | %7.2f ms
 and a "total VRAM ~X GB" line at startup. A new variant that prints these
 two lines is automatically scoreboard-compatible — add one row to RUNS.
=============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

typedef struct {
    const char *name;    // variant
    const char *mode;    // overfit | train
    const char *cmd;     // command to run
    // parsed:
    float vram;
    float nll0, nll1;            // first and last reported NLL
    float sig_mean, sig_lo, sig_hi;
    double ms_sum; int ms_cnt;   // per-step ms, excluding step 1 (warmup)
    int    steps, ok, ran;
    char   summary[3][256]; int nsum;   // "summary:" lines passed through
} Run;

static Run RUNS[] = {
    { "BBB",   "overfit", "./train_bnn_gpu overfit 200" },
    { "BBB",   "train",   "./train_bnn_gpu train 100000" },
    { "Vadam", "overfit", "./train_bnn_vadam overfit 200" },
    { "Vadam", "train",   "./train_bnn_vadam train 100000" },
    { "Ens5",  "overfit", "./train_ensemble overfit 200" },
    { "Ens5",  "train",   "./train_ensemble train 100000" },
    { "LowR4", "overfit", "./train_bnn_lowrank overfit 200" },
    { "LowR4", "train",   "./train_bnn_lowrank train 100000" },
};
static const int NRUNS = sizeof(RUNS) / sizeof(RUNS[0]);

static void execute(Run *r) {
    printf("running: %-28s ... ", r->cmd);
    fflush(stdout);

    // Every run's FULL stdout is kept. A 4-hour run must never have to be
    // repeated because a number was not printed into the summary table.
    mkdir("runs", 0755);
    char logpath[256];
    snprintf(logpath, sizeof logpath, "runs/%s_%s.log", r->name, r->mode);
    FILE *lg = fopen(logpath, "w");

    FILE *p = popen(r->cmd, "r");
    if (!p) { printf("FAILED TO LAUNCH\n"); if (lg) fclose(lg); return; }

    char line[512];
    int first = 1;
    while (fgets(line, sizeof(line), p)) {
        if (lg) fputs(line, lg);
        float v;
        if (sscanf(line, "total VRAM ~%f", &v) == 1) r->vram = v;

        if (strncmp(line, "summary:", 8) == 0 && r->nsum < 3) {
            size_t L = strcspn(line, "\n");
            if (L > 255) L = 255;
            memcpy(r->summary[r->nsum], line, L);
            r->summary[r->nsum][L] = 0;
            r->nsum++;
        }

        int step; float nll, kl, sm, lo, hi, ms;
        if (sscanf(line, "step %d | nll %f | kl %f | sigma %f [%f, %f] | %f ms",
                   &step, &nll, &kl, &sm, &lo, &hi, &ms) == 7) {
            if (first) { r->nll0 = nll; first = 0; }
            else       { r->ms_sum += ms; r->ms_cnt++; }   // skip step 1: warmup
            r->nll1 = nll; r->sig_mean = sm; r->sig_lo = lo; r->sig_hi = hi;
            r->steps = step;
        }
    }
    int rc = pclose(p);
    if (lg) fclose(lg);
    r->ran = (rc == 0 && r->steps > 0);
    if (!r->ran) { printf("FAILED (exit %d)\n", rc); return; }

    // the gate
    if (strcmp(r->mode, "overfit") == 0) r->ok = (r->nll1 < 0.05f);
    else                                 r->ok = (r->nll1 < r->nll0);

    // APPEND, never overwrite: results accumulate across sessions so old
    // configurations stay comparable to new ones.
    FILE *csv = fopen("runs/results.csv", "a");
    if (csv) {
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0)
            fprintf(csv, "utc,arm,mode,cmd,steps,ms_per_step,vram_gb,"
                         "nll_first,nll_last,sigma_mean,sigma_min,sigma_max,gate\n");
        time_t t = time(NULL); char ts[32];
        strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", gmtime(&t));
        fprintf(csv, "%s,%s,%s,\"%s\",%d,%.3f,%.3f,%.4f,%.4f,%.6f,%.6f,%.6f,%s\n",
                ts, r->name, r->mode, r->cmd, r->steps,
                r->ms_cnt ? r->ms_sum / r->ms_cnt : 0.0, r->vram,
                r->nll0, r->nll1, r->sig_mean, r->sig_lo, r->sig_hi,
                r->ok ? "PASS" : "FAIL");
        fclose(csv);
    }
    printf("done (%d steps) -> %s\n", r->steps, logpath);
}

int main(void) {
    time_t now = time(NULL);
    printf("\nBNN scorecard — %s\n", ctime(&now));   // ctime ends with \n

    for (int i = 0; i < NRUNS; i++) execute(&RUNS[i]);

    printf("\n");
    printf("variant  mode     ms/step  VRAM GB    nll first -> last    sigma mean [min, max]        gate\n");
    printf("------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < NRUNS; i++) {
        Run *r = &RUNS[i];
        if (!r->ran) {
            printf("%-8s %-8s %s\n", r->name, r->mode, "DID NOT RUN");
            continue;
        }
        printf("%-8s %-8s %7.1f  %7.2f    %7.4f -> %7.4f    %.6f [%.6f, %.6f]   %s\n",
               r->name, r->mode,
               r->ms_cnt ? r->ms_sum / r->ms_cnt : 0.0,
               r->vram, r->nll0, r->nll1,
               r->sig_mean, r->sig_lo, r->sig_hi,
               r->ok ? "PASS" : "FAIL");
    }

    // pass-through details (the ensemble's NLPD/ECE/uncertainty lines)
    for (int i = 0; i < NRUNS; i++)
        if (RUNS[i].ran && RUNS[i].nsum) {
            printf("\n%s %s:\n", RUNS[i].name, RUNS[i].mode);
            for (int j = 0; j < RUNS[i].nsum; j++)
                printf("  %s\n", RUNS[i].summary[j] + 8);
        }

    // headline ratios, from the two training runs
    Run *bbb = NULL, *vad = NULL;
    for (int i = 0; i < NRUNS; i++)
        if (strcmp(RUNS[i].mode, "train") == 0 && RUNS[i].ran) {
            if (strcmp(RUNS[i].name, "BBB") == 0)   bbb = &RUNS[i];
            if (strcmp(RUNS[i].name, "Vadam") == 0) vad = &RUNS[i];
        }
    if (bbb && vad && vad->ms_cnt && bbb->ms_cnt) {
        double bms = bbb->ms_sum / bbb->ms_cnt, vms = vad->ms_sum / vad->ms_cnt;
        printf("\nVadam vs BBB:  %.2fx faster per step  |  %.2fx less VRAM\n",
               bms / vms, bbb->vram / vad->vram);
    }

    // overall exit code: 0 only if every gate passed (usable in scripts/CI)
    int all_ok = 1;
    for (int i = 0; i < NRUNS; i++) if (!RUNS[i].ran || !RUNS[i].ok) all_ok = 0;
    printf("\noverall: %s\n", all_ok ? "ALL GATES PASS" : "SOME GATES FAIL (see table)");
    return all_ok ? 0 : 1;
}
