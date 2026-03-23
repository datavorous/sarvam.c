#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "main.h"

typedef struct {
    int dim, hidden_dim, num_layers, num_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    float* token_embedding;
    float* rms_attn;
    float *wQ, *wK, *wV, *wO;
    float* rms_ffn;
    float *w_gate, *w_down, *w_up;
    float* rms_final;
    float* wcls;
} Weights;

typedef struct {
    float *x, *xb, *xb2;
    float *q, *k, *v;
    float *hb, *hb2;
    float* att;
    float* logits;
    float *key_cache, *value_cache;
} State;

static void matmul(float* out, float* x, float* W, int n, int d)
{
    for (int i = 0; i < d; i++) {
        float val = 0;
        for (int j = 0; j < n; j++)
            val += x[j] * W[i * n + j];
        out[i] = val;
    }
}

static void rmsnorm(float* out, float* x, float* w, int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * ss * w[i];
}

static void softmax(float* x, int n)
{
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max)
            max = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

static void rope(float* x, int pos, int head_size, float theta)
{
    for (int i = 0; i < head_size; i += 2) {
        float freq = 1.0f / powf(theta, i / (float)head_size);
        float angle = pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float v0 = x[i], v1 = x[i + 1];
        x[i] = v0 * c - v1 * s;
        x[i + 1] = v0 * s + v1 * c;
    }
}

static float* load_weights(const char* path, Config* cfg)
{
    FILE* f = fopen(path, "rb");
    fread(cfg, sizeof(Config), 1, f);
    fseek(f, 0, SEEK_END);
    long size = ftell(f) - sizeof(Config);
    fseek(f, sizeof(Config), SEEK_SET);
    float* data = malloc(size);
    fread(data, 1, size, f);
    fclose(f);
    return data;
}

static void wire(Weights* w, Config* c, float* p)
{
    int L = c->num_layers;
    int kd = c->dim * c->n_kv_heads / c->num_heads;

    w->token_embedding = p;
    p += c->vocab_size * c->dim;
    w->rms_attn = p;
    p += L * c->dim;
    w->wQ = p;
    p += L * c->dim * c->dim;
    w->wK = p;
    p += L * c->dim * kd;
    w->wV = p;
    p += L * c->dim * kd;
    w->wO = p;
    p += L * c->dim * c->dim;
    w->rms_ffn = p;
    p += L * c->dim;
    w->w_gate = p;
    p += L * c->dim * c->hidden_dim;
    w->w_down = p;
    p += L * c->hidden_dim * c->dim;
    w->w_up = p;
    p += L * c->dim * c->hidden_dim;
    w->rms_final = p;
    p += c->dim;
    w->wcls = p;
}

static void alloc_state(State* s, Config* c)
{
    int kd = c->dim * c->n_kv_heads / c->num_heads;
    s->x = malloc(c->dim * sizeof(float));
    s->xb = malloc(c->dim * sizeof(float));
    s->xb2 = malloc(c->dim * sizeof(float));
    s->q = malloc(c->dim * sizeof(float));
    s->k = malloc(kd * sizeof(float));
    s->v = malloc(kd * sizeof(float));
    s->hb = malloc(c->hidden_dim * sizeof(float));
    s->hb2 = malloc(c->hidden_dim * sizeof(float));
    s->att = malloc(c->num_heads * c->seq_len * sizeof(float));
    s->logits = malloc(c->vocab_size * sizeof(float));
    s->key_cache = calloc(c->num_layers * c->seq_len * kd, sizeof(float));
    s->value_cache = calloc(c->num_layers * c->seq_len * kd, sizeof(float));
}

static float* forward(Config* cfg, Weights* w, State* s, int token, int pos)
{
    int dim = cfg->dim;
    int kd = dim * cfg->n_kv_heads / cfg->num_heads;
    int hs = dim / cfg->num_heads;
    int kv_mul = cfg->num_heads / cfg->n_kv_heads;

    memcpy(s->x, w->token_embedding + token * dim, dim * sizeof(float));

    for (int l = 0; l < cfg->num_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_attn + l * dim, dim);

        matmul(s->q, s->xb, w->wQ + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wK + l * dim * kd, dim, kd);
        matmul(s->v, s->xb, w->wV + l * dim * kd, dim, kd);

        for (int h = 0; h < cfg->num_heads; h++)
            rope(s->q + h * hs, pos, hs, 10000.0f);
        for (int h = 0; h < cfg->n_kv_heads; h++)
            rope(s->k + h * hs, pos, hs, 10000.0f);

        int loff = l * cfg->seq_len * kd;
        memcpy(s->key_cache + loff + pos * kd, s->k, kd * sizeof(float));
        memcpy(s->value_cache + loff + pos * kd, s->v, kd * sizeof(float));

        for (int h = 0; h < cfg->num_heads; h++) {
            float* q_h = s->q + h * hs;
            float* att_h = s->att + h * cfg->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k_h = s->key_cache + loff + t * kd + (h / kv_mul) * hs;
                float score = 0;
                for (int i = 0; i < hs; i++)
                    score += q_h[i] * k_h[i];
                att_h[t] = score / sqrtf(hs);
            }
            softmax(att_h, pos + 1);
            float* xb_h = s->xb + h * hs;
            memset(xb_h, 0, hs * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v_h = s->value_cache + loff + t * kd + (h / kv_mul) * hs;
                float a = att_h[t];
                for (int i = 0; i < hs; i++)
                    xb_h[i] += a * v_h[i];
            }
        }

        matmul(s->xb2, s->xb, w->wO + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++)
            s->x[i] += s->xb2[i];

        rmsnorm(s->xb, s->x, w->rms_ffn + l * dim, dim);
        matmul(s->hb, s->xb, w->w_gate + l * dim * cfg->hidden_dim, dim,
            cfg->hidden_dim);
        matmul(s->hb2, s->xb, w->w_up + l * dim * cfg->hidden_dim, dim,
            cfg->hidden_dim);
        for (int i = 0; i < cfg->hidden_dim; i++) {
            float v = s->hb[i];
            s->hb[i] = (v / (1.0f + expf(-v))) * s->hb2[i];
        }
        matmul(s->xb, s->hb, w->w_down + l * cfg->hidden_dim * dim, cfg->hidden_dim,
            dim);
        for (int i = 0; i < dim; i++)
            s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, w->rms_final, dim);
    matmul(s->logits, s->x, w->wcls, dim, cfg->vocab_size);
    return s->logits;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

int sarvam_run(const char* weights_path, const char* tokenizer_path,
    const char* prompt, int steps, int print_output,
    RunStats* stats)
{
    Config cfg;
    float* data = load_weights(weights_path, &cfg);

    Weights w;
    wire(&w, &cfg, data);
    State s;
    alloc_state(&s, &cfg);

    FILE* tf = fopen(tokenizer_path, "rb");
    fseek(tf, sizeof(int), SEEK_SET);

    char** vocab = malloc(cfg.vocab_size * sizeof(char*));
    float* scores = malloc(cfg.vocab_size * sizeof(float));
    for (int i = 0; i < cfg.vocab_size; i++) {
        int len;
        fread(&scores[i], sizeof(float), 1, tf);
        fread(&len, sizeof(int), 1, tf);
        vocab[i] = malloc(len + 1);
        fread(vocab[i], len, 1, tf);
        vocab[i][len] = '\0';
    }
    fclose(tf);

    if (steps > cfg.seq_len)
        steps = cfg.seq_len;

    int* tokens = malloc((strlen(prompt) * 4 + 8) * sizeof(int));
    int n = 0;
    tokens[n++] = 1;
    char buf[512];

    char* padded = malloc(strlen(prompt) * 3 + 4);
    int pi = 0;
    padded[pi++] = 0xE2;
    padded[pi++] = 0x96;
    padded[pi++] = 0x81;
    for (const char* c = prompt; *c; c++) {
        if (*c == ' ') {
            padded[pi++] = 0xE2;
            padded[pi++] = 0x96;
            padded[pi++] = 0x81;
        } else {
            padded[pi++] = *c;
        }
    }
    padded[pi] = '\0';

    const char* c = padded;
    while (*c) {
        int char_len = 1;
        unsigned char uc = (unsigned char)*c;
        if (uc >= 0xF0)
            char_len = 4;
        else if (uc >= 0xE0)
            char_len = 3;
        else if (uc >= 0xC0)
            char_len = 2;
        memcpy(buf, c, char_len);
        buf[char_len] = '\0';
        int id = -1;
        for (int i = 0; i < cfg.vocab_size; i++)
            if (strcmp(vocab[i], buf) == 0) {
                id = i;
                break;
            }
        if (id != -1) {
            tokens[n++] = id;
        } else {
            for (int b = 0; b < char_len; b++) {
                char hbuf[16];
                snprintf(hbuf, sizeof(hbuf), "<0x%02X>", (unsigned char)c[b]);
                int hid = -1;
                for (int i = 0; i < cfg.vocab_size; i++)
                    if (strcmp(vocab[i], hbuf) == 0) {
                        hid = i;
                        break;
                    }
                tokens[n++] = hid != -1 ? hid : 3;
            }
        }
        c += char_len;
    }
    free(padded);
    while (1) {
        float best = -1e10f;
        int best_id = -1, best_pos = -1;
        for (int i = 0; i < n - 1; i++) {
            snprintf(buf, sizeof(buf), "%s%s", vocab[tokens[i]],
                vocab[tokens[i + 1]]);
            for (int j = 0; j < cfg.vocab_size; j++)
                if (strcmp(vocab[j], buf) == 0 && scores[j] > best) {
                    best = scores[j];
                    best_id = j;
                    best_pos = i;
                }
        }
        if (best_pos == -1)
            break;
        tokens[best_pos] = best_id;
        for (int i = best_pos + 1; i < n - 1; i++)
            tokens[i] = tokens[i + 1];
        n--;
    }

    int final_tokens[4096];
    int fn = 0;
    final_tokens[fn++] = 1;
    final_tokens[fn++] = 4101;
    for (int i = 1; i < n; i++)
        final_tokens[fn++] = tokens[i];
    final_tokens[fn++] = 67476;
    final_tokens[fn++] = 4102;
    final_tokens[fn++] = 4103;
    n = fn;
    free(tokens);

    int token = final_tokens[0];
    int generated_tokens = 0;
    int steps_executed = 0;
    double inference_start = now_ms();
    double ttft_ms = -1.0;

    for (int pos = 0; pos < steps; pos++) {
        float* logits = forward(&cfg, &w, &s, pos < n ? final_tokens[pos] : token, pos);
        steps_executed++;

        int next = 0;
        for (int i = 1; i < cfg.vocab_size; i++)
            if (logits[i] > logits[next])
                next = i;

        if (pos < n - 1) {
            token = final_tokens[pos + 1];
        } else {
            if (ttft_ms < 0.0)
                ttft_ms = now_ms() - inference_start;

            if (next == 2)
                break;

            generated_tokens++;
            if (print_output) {
                const char* piece = vocab[next];
                unsigned char bv;
                if (sscanf(piece, "<0x%02hhX>", &bv) == 1) {
                    printf("%c", (char)bv);
                } else {
                    for (const char* p = piece; *p;) {
                        if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81) {
                            printf(" ");
                            p += 3;
                        } else {
                            printf("%c", *p++);
                        }
                    }
                }
                fflush(stdout);
            }

            token = next;
        }
    }

    double total_infer_ms = now_ms() - inference_start;
    if (stats) {
        stats->prompt_tokens = n;
        stats->steps_executed = steps_executed;
        stats->generated_tokens = generated_tokens;
        stats->ttft_ms = ttft_ms < 0.0 ? total_infer_ms : ttft_ms;
        stats->total_infer_ms = total_infer_ms;
    }

    if (print_output)
        printf("\n");
    return 0;
}

#ifndef SARVAM_LIB_ONLY
int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s weights.bin tokenizer.bin \"prompt\" [steps]\n",
            argv[0]);
        return 1;
    }

    int steps = argc >= 5 ? atoi(argv[4]) : 200;
    return sarvam_run(argv[1], argv[2], argv[3], steps, 1, NULL);
}
#endif