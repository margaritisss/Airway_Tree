from Registration.functions import register_affine_pairwise_parallel 

register_affine_pairwise_parallel(
    input_folder = "/projects/AirTwin_angelini/work_dataset/dataset/AIIB23/gt",
    which        = 'all',
    ref_tree     = '/home/ids/gmargari-24/airway_project/Data/Templates/template_22.nii.gz',
    output_folder= "/home/ids/gmargari-24/airway_project/Data/Affine_registered_AIB23",
    n_jobs       = 12,
    itk_threads  = 4,
)