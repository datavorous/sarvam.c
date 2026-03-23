#ifndef SARVAM_MAIN_H
#define SARVAM_MAIN_H

typedef struct {
    int prompt_tokens;
    int steps_executed;
    int generated_tokens;
    double ttft_ms;
    double total_infer_ms;
} RunStats;

int sarvam_run(const char* weights_path, const char* tokenizer_path,
    const char* prompt, int steps, int print_output,
    RunStats* stats);

#endif
