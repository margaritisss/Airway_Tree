from Registration.groupwise_registration_2_fld import register_groupwise_deformable

if __name__ == '__main__': 
    
    register_groupwise_deformable(
    input_folder_a      = "/projects/AirTwin_angelini/work_dataset/dataset/ATM22/labelsTr",
    input_folder_b      = "/projects/AirTwin_angelini/work_dataset/dataset/AIIB23/gt",
    choose              = 4,
    output_folder       = "/home/ids/gmargari-24/Data/4_8",
    groupwise_iters     = 8,
    gradient_step       = 0.2,
    blending_weight     = 0.75,
    verbose             = False,
    n_workers           = 8,  # a worker 
    threads_per_worker  = 6,  # a thread 
    template_threads    = 48, 
    monitor_interval_sec= 30,
    seed                = None
)  