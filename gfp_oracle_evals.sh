echo "" > additional_baselines_evals.txt

echo "reinforce-gfp 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-gfp/reinforce/ --tag 'score/100th' --iteration 99 >> additional_baselines_evals.txt
echo "cma-es-gfp 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-gfp/cma_es/ --tag 'score/100th' --iteration 0 >> additional_baselines_evals.txt
echo "bo-qei-gfp 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-gfp/bo_qei/ --tag 'score/100th' --iteration 20 >> additional_baselines_evals.txt

echo "reinforce-gfp 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-gfp/reinforce/ --tag 'score/50th' --iteration 99 >> additional_baselines_evals.txt
echo "cma-es-gfp 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-gfp/cma_es/ --tag 'score/50th' --iteration 0 >> additional_baselines_evals.txt
echo "bo-qei-gfp 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-gfp/bo_qei/ --tag 'score/50th' --iteration 20 >> additional_baselines_evals.txt




echo "reinforce-hopper 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-hopper/reinforce/ --tag 'score/100th' --iteration 99 >> additional_baselines_evals.txt
echo "cma-es-hopper 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-hopper/cma_es/ --tag 'score/100th' --iteration 0 >> additional_baselines_evals.txt
echo "bo-qei-hopper 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-hopper/bo_qei/ --tag 'score/100th' --iteration 20 >> additional_baselines_evals.txt

echo "reinforce-hopper 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-hopper/reinforce/ --tag 'score/50th' --iteration 99 >> additional_baselines_evals.txt
echo "cma-es-hopper 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-hopper/cma_es/ --tag 'score/50th' --iteration 0 >> additional_baselines_evals.txt
echo "bo-qei-hopper 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-hopper/bo_qei/ --tag 'score/50th' --iteration 20 >> additional_baselines_evals.txt




echo "reinforce-molecule 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-molecule/reinforce/ --tag 'score/100th' --iteration 99 >> additional_baselines_evals.txt
echo "cma-es-molecule 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-molecule/cma_es/ --tag 'score/100th' --iteration 0 >> additional_baselines_evals.txt
echo "bo-qei-molecule 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-molecule/bo_qei/ --tag 'score/100th' --iteration 20 >> additional_baselines_evals.txt

echo "reinforce-molecule 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-molecule/reinforce/ --tag 'score/50th' --iteration 99 >> additional_baselines_evals.txt
echo "cma-es-molecule 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-molecule/cma_es/ --tag 'score/50th' --iteration 0 >> additional_baselines_evals.txt
echo "bo-qei-molecule 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-molecule/bo_qei/ --tag 'score/50th' --iteration 20 >> additional_baselines_evals.txt




echo "reinforce-superconductor 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-superconductor/reinforce/ --tag 'score/100th' --iteration 99 >> additional_baselines_evals.txt
echo "cma-es-superconductor 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-superconductor/cma_es/ --tag 'score/100th' --iteration 0 >> additional_baselines_evals.txt
echo "bo-qei-superconductor 100th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-superconductor/bo_qei/ --tag 'score/100th' --iteration 20 >> additional_baselines_evals.txt

echo "reinforce-superconductor 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-superconductor/reinforce/ --tag 'score/50th' --iteration 99 >> additional_baselines_evals.txt
echo "cma-es-superconductor 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-superconductor/cma_es/ --tag 'score/50th' --iteration 0 >> additional_baselines_evals.txt
echo "bo-qei-superconductor 50th" >> additional_baselines_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-superconductor/bo_qei/ --tag 'score/50th' --iteration 20 >> additional_baselines_evals.txt