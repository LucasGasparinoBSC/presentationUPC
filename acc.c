#include <openacc.h>
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

int main(void)
{
    int n = 512*512;
    int iters = 10;

    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *c = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Arrays a, b and c are created on the device;
    // a and b are copied to device before k=0 loop;
    // c is copied back to host after k=99 loop.
    // Data is destroyed on the device after the loop.
    PUSH_RANGE("Kernel 1", 0);
    #pragma acc data copyin(a[0:n], b[0:n]) copyout(c[0:n])
    {
        for (int k = 0; k < iters; k++)
        {
            PUSH_RANGE("Kernel 1 loop", k);
            #pragma acc parallel loop
            for (int i = 0; i < n; i++) {
                c[i] = a[i] + b[i];
            }
            POP_RANGE;
        }
    }
    POP_RANGE;

    // Arrays a, b and c are created on the device at k=0 loop;
    // a and b are copied to device before every k loop;
    // c is copied back to host after every k loop;
    PUSH_RANGE("Kernel 2", 1);
    for (int k = 0; k < iters; k++)
    {
        PUSH_RANGE("Kernel 2 loop", k);
        #pragma acc data copyin(a[0:n], b[0:n]) copyout(c[0:n])
        {
            #pragma acc parallel loop
            for (int i = 0; i < n; i++) {
                c[i] = a[i] + b[i];
            }
        }
        POP_RANGE;
    }
    POP_RANGE;

    // Arrays a, b and c are created on the device;
    // a and b are copied to device before k=0 loop;
    // c is copied back to host after k=99 loop;
    // Data is destroyed on the device at the exit data clause;
    PUSH_RANGE("Kernel 3", 2);
    #pragma acc enter data copyin(a[0:n], b[0:n]) create(c[0:n])
    for (int k = 0; k < iters; k++)
    {
        PUSH_RANGE("Kernel 3 loop", k);
        #pragma acc parallel loop
        for (int i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
        POP_RANGE;
    }
    #pragma acc exit data copyout(c[0:n])
    POP_RANGE;

    return EXIT_SUCCESS;
}