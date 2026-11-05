@echo off
rem Post-hoc external comparator: Skeleton Recall loss, three seeds, leak-free test and OOD sets.
rem Run from the repository root; finished runs are skipped and interrupted ones resume.
setlocal enabledelayedexpansion
set PYTHONPATH=src
set C=baseline_skelrecall
for %%S in (17 29 43) do (
  if exist outputs\%C%\seed_%%S\evaluation_test\summary.json (
    echo --- skip %C% seed %%S
  ) else (
    echo === %C% seed %%S  !date! !time!
    python scripts\train.py --config configs\%C%.yaml --seed %%S --final --resume outputs\%C%\seed_%%S\last.pt
    python scripts\evaluate.py --config configs\%C%.yaml --seed %%S --split test
  )
  if not exist outputs\%C%\seed_%%S\evaluation_ood\summary.json (
    python scripts\evaluate.py --config configs\%C%.yaml --seed %%S --split ood
  )
)
echo skeleton recall queue finished !date! !time!
endlocal
