design-baselines plot --dir ~/final-results/online/gradient-ascent-hopper/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Average Return' --tag 'score/100th'
design-baselines plot --dir ~/final-results/online/gradient-ascent-hopper/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Predicted Average Return' --tag 'oracle/min_of_mean/mean'

design-baselines plot --dir ~/final-results/online/gradient-ascent-superconductor/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Critical Temperature' --tag 'score/100th'
design-baselines plot --dir ~/final-results/online/gradient-ascent-superconductor/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Predicted Critical Temperature' --tag 'oracle/min_of_mean/mean'

design-baselines plot --dir ~/final-results/online/gradient-ascent-ant/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Average Return' --tag 'score/100th'
design-baselines plot --dir ~/final-results/online/gradient-ascent-ant/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Predicted Average Return' --tag 'oracle/min_of_mean/mean'

design-baselines plot --dir ~/final-results/online/gradient-ascent-dkitty/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Average Return' --tag 'score/100th'
design-baselines plot --dir ~/final-results/online/gradient-ascent-dkitty/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Predicted Average Return' --tag 'oracle/min_of_mean/mean'

design-baselines plot --dir ~/final-results/online/gradient-ascent-gfp/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Protein Fluorescence' --tag 'score/100th'
design-baselines plot --dir ~/final-results/online/gradient-ascent-gfp/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Predicted Protein Fluorescence' --tag 'oracle/min_of_mean/mean'

design-baselines plot --dir ~/final-results/online/gradient-ascent-molecule/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Drug Activity' --tag 'score/100th'
design-baselines plot --dir ~/final-results/online/gradient-ascent-molecule/gradient_ascent/ --xlabel 'Gradient Ascent Steps' --ylabel 'Predicted Drug Activity' --tag 'oracle/min_of_mean/mean'