import argparse as ap
import os


if __name__ == "__main__":

    parser = ap.ArgumentParser("Generate MBO Scripts")

    parser.add_argument("--image-dir", type=str,
                        default='/global/scratch/btrabucco/mbo.img')
    parser.add_argument("--scripts-dir", type=str,
                        default='/global/scratch/btrabucco/scripts')
    parser.add_argument("--results-dir", type=str,
                        default='/global/scratch/btrabucco/mbo-results')

    parser.add_argument("--slurm-account", type=str, default='co_rail')
    parser.add_argument("--slurm-partition", type=str, default='savio3_gpu')
    parser.add_argument("--slurm-qos", type=str, default='rail_gpu3_normal')
    parser.add_argument("--slurm-hours", type=int, default=24)
    parser.add_argument("--slurm-memory", type=int, default=80)

    parser.add_argument("--singularity-args", type=str, default="")

    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--num-parallel", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)

    args = parser.parse_args()

    os.makedirs(args.scripts_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    for baseline in ["autofocused-cbas",
                     "bo-qei",
                     "cbas",
                     "cma-es",
                     "gradient-ascent",
                     "gradient-ascent-min-ensemble",
                     "gradient-ascent-mean-ensemble",
                     "mins",
                     "reinforce"]:

        for task in ["tf-bind-10",
                     "nas"]:

            run_script = f"""#!/bin/bash
. /packages/anaconda3/etc/profile.d/conda.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200_linux/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro131/bin

conda activate design-baselines

{baseline} {task} \
--local-dir {args.results_dir}/{baseline}-{task} \
--cpus {args.cpus} \
--gpus {args.gpus} \
--num-parallel {args.num_parallel} \
--num-samples {args.num_samples}"""

            with open(f"{args.scripts_dir}/"
                      f"run_{baseline}_{task}.sh", "w") as f:
                f.write(run_script)

            launch_script = f"""#!/bin/bash
#SBATCH --job-name={baseline}-{task}
#SBATCH --account={args.slurm_account}
#SBATCH --time={args.slurm_hours}:00:00
#SBATCH --partition={args.slurm_partition}
#SBATCH --qos={args.slurm_qos}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.slurm_memory}G
#SBATCH --gres=gpu:TITAN:{args.gpus}

singularity exec --nv -w {args.singularity_args} \
    {args.image_dir} \
    /bin/bash \
    {args.scripts_dir}/run_{baseline}_{task}.sh"""

            with open(f"{args.scripts_dir}/"
                      f"launch_{baseline}_{task}.sh", "w") as f:
                f.write(launch_script)

            print(f"sbatch {args.scripts_dir}/launch_{baseline}_{task}.sh")
