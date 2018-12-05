import os
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()


print("MPI is running with local rank: {} rank: {} in total of {} processes." \
          .format(hvd.local_rank(), hvd.rank(), hvd.size()))