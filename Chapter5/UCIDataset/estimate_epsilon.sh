# Statistcal Estimation Runner
for eps in 0.025 0.05 0.75
do
    for (( INNUM=0; INNUM<=10; INNUM++ ))
    do

	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset boston1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset energy1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset yacht1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset concrete1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset powerplant1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset wine1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset kin8nm1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset naval1 --eps $eps &
	I=$(( INNUM+11  ))
	python3 EstimateStatisticalProperties.py --imnum $I --dataset boston1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $I --dataset energy1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $I --dataset yacht1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $I --dataset concrete1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $I --dataset powerplant1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $I --dataset wine1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $I --dataset kin8nm1 --eps $eps &
	python3 EstimateStatisticalProperties.py --imnum $I --dataset naval1 --eps $eps 
    done
done
