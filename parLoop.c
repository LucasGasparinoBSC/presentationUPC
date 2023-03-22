void kernel()
{
    #pragma acc parallel loop gang // thread block partition
    for (int i = 0; i < n; i++) {
        ...
        #pragma acc loop worker // Direction y of the thread block
        for (int j = 0; j < n; j++) {
            ...
            #pragma acc loop vector // Direction x of the thread block
            for (int k = 0; k < n; k++) {
                ...
            }
        }
    }
}
