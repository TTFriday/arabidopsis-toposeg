@echo off
rem Diagnostic evaluations: the published nominal test assignment and the unmapped OOD set.
rem Safe to re-run; a run with an existing summary.json is skipped. Do not edit while executing.
setlocal enabledelayedexpansion
set PYTHONPATH=src
for %%P in (nominal ood) do (
  for %%C in (baseline_region baseline_cldice full ablation_uniform ablation_no_joint_root ablation_no_skeleton_head) do (
    for %%S in (17 29 43) do (
      if exist outputs\%%C\seed_%%S\evaluation_%%P\summary.json (
        echo --- skip %%C seed %%S %%P
      ) else (
        echo === %%C seed %%S %%P  !date! !time!
        python scripts\evaluate.py --config configs\%%C.yaml --seed %%S --split %%P
      )
    )
  )
)
echo protocol evaluations finished !date! !time!
endlocal
