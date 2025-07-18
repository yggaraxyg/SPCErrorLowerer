#include<iostream>
#include<cmath>
using namespace std;

double getError(int lumens, double val);
double findBestInGroup(int lumens, double min, double max, int sections, int MaxLoops, int Loops, double & ErAt);

int main(){
  final int LUMEN_MIN = 0;
  final int LUMEN_MAX = 10000;
  final int LUMEN_INT = 1;

  final int DIST_NUM_MIN = 0;
  final int DIST_NUM_MAX = 10;
  final int DIST_NUM_INT = 1;

  final double DIST_MIN_MIN = 0;
  final double DIST_MIN_MAX = 1;
  final double DIST_MIN_INT = 0.01;
  
  final double DIST_MAX_MIN = 0;
  final double DIST_MAX_MAX = 1;
  final double DIST_MAX_INT = 0.01;

  final int NUMBER_OF_RUNS_MIN = 1;
  final int NUMBER_OF_RUNS_MAX = 100;
  final int NUMBER_OF_RUNS_INT = 1;
  
  final int LASER_REPETITION_PERIOD_MIN = 1;
  final int LASER_REPETITION_PERIOD_MAX = 10000;
  final int LASER_REPETITION_PERIOD_INT = 1;

  final int LASER_TIME_PERIOD_BINS_MIN = 1;
  final int LASER_TIME_PERIOD_BINS_MAX = 10000;
  final int LASER_TIME_PERIOD_BINS_INT = 1;

  final int LASER_FWHM_MIN = 1;
  final int LASER_FWHM_MAX = 10;
  final int LASER_FWHM_INT = 1;
  
  final int LUMEN_BOX = (int)((LUMEN_MAX-LUMEN_MIN)/LUMEN_INT);
  final int DIST_NUM_BOX = (int)((DIST_NUM_MAX-DIST_NUM_MIN)/DIST_NUM_INT);
  final int DIST_MIN_BOX = (int)((DIST_MIN_MAX-DIST_MIN_MIN)/DIST_MIN_INT);
  final int DIST_MAX_BOX = (int)((DIST_MAX_MAX-DIST_MAX_MIN)/DIST_MAX_INT);
  final int NUMBER_OF_RUNS_BOX = (int)((NUMBER_OF_RUNS_MAX-NUMBER_OF_RUNS_MIN)/NUMBER_OF_RUNS_INT);
  final int LASER_REPETITION_PERIOD_BOX = (int)((LASER_REPETITION_PERIOD_MAX-LASER_REPETITION_PERIOD_MIN)/LASER_REPETITION_PERIOD_INT);
  final int LASER_TIME_PERIOD_BINS_BOX = (int)((LASER_TIME_PERIOD_BINS_MAX-LASER_TIME_PERIOD_BINS_MIN)/LASER_TIME_PERIOD_BINS_INT);
  final int LASER_FWHM_BOX = (int)((LASER_FWHM_MAX-LASER_FWHM_MIN)/LASER_FWHM_INT);
  
  double******** optimalBlockage = new double*******[LUMEN_BOX];
  double******** errorAtOptimal = new double*******[LUMEN_BOX];
  for(int LUMEN=0; LUMEN<LUMEN_BOX; LUMEN++){
    double******* optimalBlockage = new double******[DIST_NUM_BOX];
    double******* errorAtOptimal = new double******[DIST_NUM_BOX];
    for(int DIST_NUM=0; DIST_NUM<DIST_NUM_BOX; DIST_NUM++){
      double****** optimalBlockage = new double*****[DIST_MIN_BOX];
      double****** errorAtOptimal = new double*****[DIST_MIN_BOX];
      for(int DIST_MIN=0; DIST_MIN<DIST_MIN_BOX; DIST_MIN++){
	double***** optimalBlockage = new double****[DIST_MAX_BOX];
	double***** errorAtOptimal = new double****[DIST_MAX_BOX];
	for(int DIST_MAX=0; DIST_MAX<DIST_MAX_BOX; DIST_MAX++){
	  double**** optimalBlockage = new double***[NUMBER_OF_RUNS_BOX];
	  double**** errorAtOptimal = new double***[NUMBER_OF_RUNS_BOX];
	  for(int NUMBER_OF_RUNS=0; NUMBER_OF_RUNS<NUMBER_OF_RUNS_BOX; NUMBER_OF_RUNS++){
	    double*** optimalBlockage = new double**[LASER_REPETITION_PERIOD_BOX];
	    double*** errorAtOptimal = new double**[LASER_REPETITION_PERIOD_BOX];
	    for(int LASER_REPETITION_PERIOD=0; LASER_REPETITION_PERIOD<LASER_REPETITION_PERIOD_BOX; LASER_REPETITION_PERIOD++){
	      double** optimalBlockage = new double*[LASER_TIME_PERIOD_BINS_BOX];
	      double** errorAtOptimal = new double*[LASER_TIME_PERIOD_BINS_BOX];
	      for(int LASER_TIME_PERIOD_BINS=0; LASER_TIME_PERIOD_BINS<LASER_TIME_PERIOD_BINS_BOX; LASER_TIME_PERIOD_BINS++){
		double* optimalBlockage = new double[LASER_FWHM_BOX];
		double* errorAtOptimal = new double[LASER_FWHM_BOX];
	      } 
	    } 
	  } 
	} 
      } 
    }
  }


  for(int LUMEN=0; LUMEN<LUMEN_BOX; LUMEN++){
    for(int DIST_NUM=0; DIST_NUM<DIST_NUM_BOX; DIST_NUM++){
      for(int DIST_MIN=0; DIST_MIN<DIST_MIN_BOX; DIST_MIN++){
	for(int DIST_MAX=0; DIST_MAX<DIST_MAX_BOX; DIST_MAX++){
	  for(int NUMBER_OF_RUNS=0; NUMBER_OF_RUNS<NUMBER_OF_RUNS_BOX; NUMBER_OF_RUNS++){
	    for(int LASER_REPETITION_PERIOD=0; LASER_REPETITION_PERIOD<LASER_REPETITION_PERIOD_BOX; LASER_REPETITION_PERIOD++){
	      for(int LASER_TIME_PERIOD_BINS=0; LASER_TIME_PERIOD_BINS<LASER_TIME_PERIOD_BINS_BOX; LASER_TIME_PERIOD_BINS++){
		for(int LASER_FWHM=0; LASER_FWHM<LASER_FWHM_BOX; LASER_FWHM++){
		  double ErAt=0;
		  optimalBlockage[LUMEN][DIST_NUM][DIST_MIN][DIST_MAX][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM] = findBestInGroup( (LUMEN_MIN+(LUMEN_INT*LUMEN)),
																				       (DIST_NUM_MIN+(DIST_NUM_INT*DIST_NUM)),
																				       (DIST_MIN_MIN+(DIST_MIN_INT*DIST_MIN)),
																				       (DIST_MAX_MIN+(DIST_MAX_INT*DIST_MAX)),
																				       (NUMBER_OF_RUNS_MIN+(NUMBER_OF_RUNS_INT*NUMBER_OF_RUNS)),
																				       (LASER_REPETITION_PERIOD_MIN+(LASER_REPETITION_PERIOD_INT*LASER_REPETITION_PERIOD)),
																				       (LASER_TIME_PERIOD_BINS_MIN+(LASER_TIME_PERIOD_BINS_INT*LASER_TIME_PERIOD_BINS)),
																				       (LASER_FWHM_MIN+(LASER_FWHM_INT*LASER_FWHM)),
																				       0, 100, 10, 6, 0, ErAt);
		  errorAtOptimal[LUMEN][DIST_NUM][DIST_MIN][DIST_MAX][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM] = ErAt;
		  cout<<"LUMEN: "<<(LUMEN_MIN+(LUMEN_INT*LUMEN))
		      <<" DIST NUM: "<<(DIST_NUM_MIN+(DIST_NUM_INT*DIST_NUM))
		      <<" DIST MIN: "<<(DIST_MIN_MIN+(DIST_MIN_INT*DIST_MIN))
		      <<" DIST MAX: "<<(DIST_MAX_MIN+(DIST_MAX_INT*DIST_MAX))
		      <<" NUMBER OF RUNS: "<<(NUMBER_OF_RUNS_MIN+(NUMBER_OF_RUNS_INT*NUMBER_OF_RUNS))
		      <<" LASER REPETITION PERIOD: "<<(LASER_REPETITION_PERIOD_MIN+(LASER_REPETITION_PERIOD_INT*LASER_REPETITION_PERIOD))
		      <<" LASER TIME PERIOD BINS: "<<(LASER_TIME_PERIOD_BINS_MIN+(LASER_TIME_PERIOD_BINS_INT*LASER_TIME_PERIOD_BINS))
		      <<" LASER FWHM: "<<(LASER_FWHM_MIN+(LASER_FWHM_INT*LASER_FWHM))
		      <<" Optimal Blockage: "<<optimalBlockage[LUMEN][DIST_NUM][DIST_MIN][DIST_MAX][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM]
		      <<" Error at Optimal: "<<errorAtOptimal[LUMEN][DIST_NUM][DIST_MIN][DIST_MAX][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM]<<"."<<endl;
		} 
	      } 
	    } 
	  } 
	} 
      } 
    }
  }
  
  return 0;
}

double getError(int LUMEN, int DIST_NUM, double DIST_MIN, double DIST_MAX, int NUMBER_OF_RUNS, int LASER_REPETITION_PERIOD, int LASER_TIME_PERIOD_BINS, int LASWER_FWHM, double val){
  return val;
  //this is a placeholder function to determine the error value. I will find it out using the library you mentioned but in the meantime it will just do nothing.
}

double findBestInGroup(int LUMEN, int DIST_NUM, double DIST_MIN, double DIST_MAX, int NUMBER_OF_RUNS, int LASER_REPETITION_PERIOD, int LASER_TIME_PERIOD_BINS, int LASWER_FWHM, double min, double max, int sections, int MaxLoops, int Loops, double & ErAt){
  //this is a rescursive function meant to figure out what value between Max and Min gives the minimum error. it does this by divinig the area between max and min into evenly-spaced sections, finding the error of the middle point in each section, finding the point with the lowest error, taking that point's lowest-error neighbor (neighbor meaning point with value already determined one section before or after one point), and performing the sampe operation with those two points as max and min (determined by whichever is lower and higher respectively) then recursing until desired number of recursions are performed. once desired number is perfromed, we take the midpoint of the area beteween max and min and call it sufficient.
  cout<<"Recursion Number: "<<loops<<" Min: "<<min<<" Max: "<<max<<endl;
  if(maxLoops<=Loops){
    ErAt = getError(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, ((min+max)/2));
    return ((min+max)/2);
  }
  Loops++;
  double newMin=min;
  double newMax=max;
  double* ErrorOfSection  = new double[sections];
  double* ValOfSection = new double[sections];
  int mindex=0;
  for(int i=0; i<sections; i++){
    ValOfSection[i]= (((max-min)/sections)*(i+0.5))+min;
    double avg = getError(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, ValOfSection[i]);
    for(int j=1; j<=5; j++){
      avg+=getError(ValOfSection[i]+(j*((max-min)/pow(sections,2))));
      avg+=getError(ValOfSection[i]-(j*((max-min)/pow(sections,2))));
    }
    avg=avg/11
    ErrorOfSection[i] = avg;
    if(ErrorOfSection[i]<ErrorOfSection[mindex]){
      mindex=i;
    }
  }
  
  if(mindex==0){
    newMin=ValOfSection[0];
    newMax=ValOfSection[1];
  }else if(mindex==sections-1){
    newMin=ValOfSection[sections-2];
    newMax=ValOfSection[sections-1];
  }else{
    if(ErrorOfSection[mindex-1] < ErrorOfSection [mindex+1]){
      newMin=ValOfSection[mindex-1];
      newMax=ValOfSection[mindex];
    }else{
      newMin=ValOfSection[mindex];
      newMax=ValOfSection[mindex+1];
    }
  }
  delete[] ErrorOfSection;
  delete[] ValOfSection;
  return findBestInGroup(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, newMin, newMax, sections, MaxLoops, Loops+1, ErAt);
}
