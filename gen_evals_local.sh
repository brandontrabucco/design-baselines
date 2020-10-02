echo "" > gradient_ascent_evals.txt
echo "" > coms_evals.txt

design-baselines evaluate-offline --dir ~/final-results/online/gradient-ascent-hopper/gradient_ascent/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/gradient-ascent-superconductor/gradient_ascent/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/gradient-ascent-ant/gradient_ascent/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/gradient-ascent-dkitty/gradient_ascent/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/gradient-ascent-gfp/gradient_ascent/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/gradient-ascent-molecule/gradient_ascent/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> gradient_ascent_evals.txt

design-baselines evaluate-offline --dir ~/final-results/online/online-hopper/online/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/online-superconductor/online/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/online-ant/online/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/online-dkitty/online/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/online-gfp/online/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-results/online/online-molecule/online/ --tag 'score/100th' --eval-tag 'oracle/min_of_mean/mean' >> coms_evals.txt


echo "" > gradient_ascent_evals.txt
echo "" > coms_evals.txt

design-baselines evaluate-offline --dir ~/final-offline/final-gradient-ascent-hopper/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-gradient-ascent-superconductor/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-gradient-ascent-ant/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-gradient-ascent-dkitty/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-gradient-ascent-gfp/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-gradient-ascent-molecule/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt

design-baselines evaluate-offline --dir ~/final-offline/final-csm-hopper/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-csm-superconductor/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-csm-ant/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-csm-dkitty/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-csm-gfp/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline --dir ~/final-offline/final-csm-molecule/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt


echo "" > gradient_ascent_evals.txt
echo "" > coms_evals.txt

design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-gradient-ascent-hopper/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-gradient-ascent-superconductor/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-gradient-ascent-ant/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-gradient-ascent-dkitty/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-gradient-ascent-gfp/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-gradient-ascent-molecule/gradient_ascent/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> gradient_ascent_evals.txt

design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-csm-hopper/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-csm-superconductor/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-csm-ant/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-csm-dkitty/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-csm-gfp/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt
design-baselines evaluate-offline-per-seed --dir ~/final-offline/final-csm-molecule/csm/ --tag 'score/100th' --eval-tag 'held_out/min_of_mean/mean' >> coms_evals.txt