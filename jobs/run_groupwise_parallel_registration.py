from Registration.groupwise_parallel_registration import register_groupwise_deformable

if __name__ == '__main__': 
    
    register_groupwise_deformable(
        input_folder           = '/home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Affine_registered/ATM22',
        output_folder          = '/home/ids/gmargari-24/airway_project/Data/Registered_on_Template_22_23/Groupwise_non_linear_registered/ATM22',
        existing_template_path = '/home/ids/gmargari-24/airway_project/Data/Templates/template_22_23.nii.gz',
        n_workers              = 12,
        threads_per_worker     = 4,
        verbose                = False,
    )