from airway_project.Registration.groupwise_registration_2_fld import register_groupwise_deformable

if __name__ == '__main__': 
    
    register_groupwise_deformable(
    input_folder_a      = "/home/ids/gmargari-24/Data/Affine_registered_ATM22",
    input_folder_b      = None,
    choose              = 24,
    output_folder       = "/home/ids/gmargari-24/Data/Template",
    groupwise_iters     = 4,
    gradient_step       = 0.2,
    blending_weight     = 0.75,
    verbose             = False,
    n_workers           = 8,  # a worker 
    threads_per_worker  = 6,  # a thread 
    template_threads    = 48, 
    monitor_interval_sec= 30,
    seed                = None
)  