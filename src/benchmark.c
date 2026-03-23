#include "main.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s weights.bin tokenizer.bin \"prompt\" [steps]\n",
            argv[0]);
        return 1;
    }

    int steps = argc >= 5 ? atoi(argv[4]) : 200;
    RunStats stats;
    int rc = sarvam_run(argv[1], argv[2], argv[3], steps, 0, &stats);
    if (rc != 0)
        return rc;

    double ms_per_token = stats.generated_tokens > 0
        ? stats.total_infer_ms / (double)stats.generated_tokens
        : 0.0;
    double tok_per_s = ms_per_token > 0.0 ? 1000.0 / ms_per_token : 0.0;

    printf("TBT_ms: %.3f\n", ms_per_token);
    printf("Throughput_tok_per_s: %.3f\n", tok_per_s);
    printf("TotalTime_ms: %.3f\n", stats.total_infer_ms);
    return 0;
}
