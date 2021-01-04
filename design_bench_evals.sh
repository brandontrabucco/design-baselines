echo "" > gfp_oracle_evals.txt

echo "autofocus-gfp-v1 100th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/autofocus-gfp-v1/autofocus/ --tag 'score/100th' --iteration 50 --confidence >> gfp_oracle_evals.txt
echo "cbas-gfp-v1 100th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cbas-gfp-v1/cbas/ --tag 'score/100th' --iteration 50 --confidence >> gfp_oracle_evals.txt
echo "mins-gfp-v1 100th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/mins-gfp-v1/mins/ --tag 'exploitation/actual_ys/100th' --iteration 0 --confidence >> gfp_oracle_evals.txt

echo "grad-gfp-v1 100th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/grad-gfp-v1/gradient_ascent/ --tag 'score/100th' --iteration 200 --confidence >> gfp_oracle_evals.txt
echo "grad-gfp-v1-min-ensemble 100th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/grad-gfp-v1-min-ensemble/gradient_ascent/ --tag 'score/100th' --iteration 200 --confidence >> gfp_oracle_evals.txt
echo "grad-gfp-v1-mean-ensemble 100th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/grad-gfp-v1-mean-ensemble/gradient_ascent/ --tag 'score/100th' --iteration 200 --confidence >> gfp_oracle_evals.txt

echo "reinforce-gfp-v1 100th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-gfp-v1/reinforce/ --tag 'score/100th' --iteration 99 --confidence >> gfp_oracle_evals.txt
echo "cma-es-gfp-v1 100th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-gfp-v1/cma_es/ --tag 'score/100th' --iteration 0 --confidence >> gfp_oracle_evals.txt
echo "bo-qei-gfp-v1 100th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-gfp-v1/bo_qei/ --tag 'score/100th' --iteration 20 --confidence >> gfp_oracle_evals.txt





echo "autofocus-gfp-v1 50th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/autofocus-gfp-v1/autofocus/ --tag 'score/50th' --iteration 50 --confidence >> gfp_oracle_evals.txt
echo "cbas-gfp-v1 50th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cbas-gfp-v1/cbas/ --tag 'score/50th' --iteration 50 --confidence >> gfp_oracle_evals.txt
echo "mins-gfp-v1 50th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/mins-gfp-v1/mins/ --tag 'exploitation/actual_ys/50th' --iteration 0 --confidence >> gfp_oracle_evals.txt

echo "grad-gfp-v1 50th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/grad-gfp-v1/gradient_ascent/ --tag 'score/50th' --iteration 200 --confidence >> gfp_oracle_evals.txt
echo "grad-gfp-v1-min-ensemble 50th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/grad-gfp-v1-min-ensemble/gradient_ascent/ --tag 'score/50th' --iteration 200 --confidence >> gfp_oracle_evals.txt
echo "grad-gfp-v1-mean-ensemble 50th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/grad-gfp-v1-mean-ensemble/gradient_ascent/ --tag 'score/50th' --iteration 200 --confidence >> gfp_oracle_evals.txt

echo "reinforce-gfp-v1 50th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/reinforce-gfp-v1/reinforce/ --tag 'score/50th' --iteration 99 --confidence >> gfp_oracle_evals.txt
echo "cma-es-gfp-v1 50th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/cma-es-gfp-v1/cma_es/ --tag 'score/50th' --iteration 0 --confidence >> gfp_oracle_evals.txt
echo "bo-qei-gfp-v1 50th" >> gfp_oracle_evals.txt
design-baselines evaluate-fixed --dir ~/rebuttal/bo-qei-gfp-v1/bo_qei/ --tag 'score/50th' --iteration 20 --confidence >> gfp_oracle_evals.txt


