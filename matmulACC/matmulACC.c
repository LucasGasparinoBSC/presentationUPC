#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <nvToolsExt.h>

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();

void matmulKernels(int n, float *a, float *b, float *c) {
    int i, j, k;
    float sum;
    #pragma acc kernels
    {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                sum = 0.0;
                for (k = 0; k < n; k++) {
                    sum += a[i*n+k] * b[k*n+j];
                }
                c[i*n+j] = sum;
            }
        }
    }
}

void matmulParallel_V0(int n, float *a, float *b, float *c) {
    int i, j, k;
    float sum;
    #pragma acc parallel loop gang worker collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            #pragma acc loop vector reduction(+:sum)
            for (k = 0; k < n; k++) {
                sum += a[i*n+k] * b[k*n+j];
            }
            c[i*n+j] = sum;
        }
    }
}

int main()
{
    int n = 2000;

    float *a = (float*)malloc(n*n*sizeof(float));
    float *b = (float*)malloc(n*n*sizeof(float));
    float *c = (float*)malloc(n*n*sizeof(float));

    #pragma acc parallel loop
    for (int i = 0; i < n*n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    PUSH_RANGE("matmulKernels", 0);
    for (int i = 0; i < 2; i++) {
        PUSH_RANGE("iter", i)
        matmulKernels(n, a, b, c);
        POP_RANGE
    }
    POP_RANGE;
    printf("c[0] = %f\n", c[0]);

    PUSH_RANGE("matmulKernels", 0);
    for (int i = 0; i < 10; i++) {
        PUSH_RANGE("iter", i)
        matmulParallel_V0(n, a, b, c);
        POP_RANGE
    }
    POP_RANGE;
    printf("c[0] = %f\n", c[0]);

    return 0;
}