#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <ctype.h>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
using namespace std;

int main(){
  for (int test = 0; test < 10; test++){
    for (int testN = 0; testN < 10; testN++){      
      /// Parameters
      
      double mu = 2.0; 
      double alpha = 1.0*test;
      double beta = 0.3;
      double sigma = 0.01;
      int Ninit = 100;
      double Mrate1 = 0.5;
      double Mrate2 = 4.5;
      double Mrate3 = 4.5;
      double gamma = 1.0;
      double delta = 0.05;

      /// For generating Network
      static int N0 = 50000;
      static double P1[50000][2];
      static int AK[50000][200];
      static int UK [50000];
      static int UKRef [50000];
      static int DK[50000];
      static int Prob[50000];
      static int SK[50000];
      static int AKRef[50000][200];
      static double Rate[50000][200];
      static int SAK [13][50000];
      static int S [13]; 
      static double Par [13]; 

      int Nstates = 13;
      int Ninfectious = 4; /// This means the 5'th state
      /// For simulating Actual Disease
      int NExp = 0; int NInf = 0;
      srand (time(0));
      
      double ra,ra1,ra2,ran1,ran2,nran1,nran2; int id1,id2,accra; int NRe = N0;double r;
      clock_t begin = clock();

      for (int i = 0; i < Nstates-1; i++){
	S[i] = 0; 
	if (i < 4){Par[i] = Mrate1;}
	else if (i < 8){Par[i] = Mrate2;}
	else{	  Par[i] = Mrate3;}
      }  
      Par[12] = 0;
      /// Here we initialize the system
      int c = 0; int acc = 0;
      while (c<N0){
	UK[c] = 0;    DK[c] = 0;    UKRef[c] = 0; SK[c] = -1;
	acc = 0;
	while (acc == 0){
	  ra1 = double(rand()/double(RAND_MAX)); 	  ra2 = double(rand()/double(RAND_MAX)); 
	  if(ra2 < exp(-gamma*ra1)){ acc = 1;}}
	Prob[c] = ra1 + delta;

	for (int j = 0; j<200; j++){
	  AK[c][j] = -1;      AKRef[c][j] = -1;      Rate[c][j] = -1;
	  if (j < 13){
	    SAK[j][c] = -1;      }    }
	acc = 0;
	while (acc==0){
	  P1[c][0] = -1.0 + 2*double(rand()/double(RAND_MAX));
	  P1[c][1] = -1.0 + 2*double(rand()/double(RAND_MAX));
	  if (sqrt(P1[c][0]*P1[c][0] + P1[c][1]*P1[c][1]) < 1.0){
	    acc = 1;
	  }    }
	c++;
      }
      ///////////////////////////////////////////////////////////////////      
      cout << "Here " << endl;
                  
      c = 0; 
  /// Here we construct and connect network ///
  while (c < mu*NRe){
    accra = 0;
    while (accra == 0){
      ra1 = double(rand()/double(RAND_MAX)); if (ra1 == 1){ra1 = 0.999999;}
      ra2 = double(rand()/double(RAND_MAX)); if (ra2 == 1){ra2 = 0.999999;}
      id1 = floor(NRe*ra1);     id2 = floor(NRe*ra2);
      ra1 = double(rand()/double(RAND_MAX)); if (ra1 == 1){ra1 = 0.999999;}
      ra2 = double(rand()/double(RAND_MAX)); if (ra2 == 1){ra2 = 0.999999;}
      if (ra1 < Prob[id1] && ra2 < Prob[id2] && UK[id1] < 200 && UK[id2] < 200){
        accra = 1;
      }
    }
    acc = 1;
    for (int i1 = 0; i1 < UK[id1]; i1++){ // Make sure no element is present twice
      if (AK[id1][i1] == id2){	acc = 0;      }
    }
    if ((id1 != id2) && acc == 1){ // Assign contacts and rates
      r = sqrt( pow((P1[id1][0] - P1[id2][0]),2) + pow((P1[id1][1] - P1[id2][1]),2) );
      ra = double(rand()/double(RAND_MAX));
      if ( exp(-alpha*r) > ra ){
	ran1=double(rand()/double(RAND_MAX)); ran2=double(rand()/double(RAND_MAX));
	nran1 = cos(2*3.14*ran2)*sqrt(-2.*log(ran1)); nran2 = sin(2*3.14*ran2)*sqrt(-2.*log(ran1));

	AK[id1][UK[id1]] = id2;	        AKRef[id1][UK[id1]] = id2;
	Rate[id1][UK[id1]] = beta + sigma*nran1;

	AK[id2][UK[id2]] = id1; 	AKRef[id2][UK[id2]] = id1;
	Rate[id2][UK[id2]] = beta + sigma*nran1;

	UK[id1]++; UK[id2]++; 
	UKRef[id1]++; UKRef[id2]++; 
	c++;
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
    
  clock_t end = clock();  double elape = double(end - begin) / CLOCKS_PER_SEC;  cout << elape << endl;

  //////////////////////////////////////////// ///////////////////// /////////////////////// /////////////////////// /////////////////////////////////////////

  int Af,B,cin1,cjn1,ckn1,idy;
  int idx = 0;   int on = 1;  int cac;  int AC = 0;
  double Tot = 0;  double TotMov = 0; double TotInf = 0;  int NRecov = 0;
  int cinf = 0; int cupd = 0; int clickN = 0;
  double nts = 0.1; int click = 0;
  c = 0;  double Csum = 0; double RT = 0; double dt; double exI,exM;
  cout << "START!!!    " << endl;
  clock_t begin1 = clock();

  /// Now make initial infectious
  for (int iin = 0; iin < Ninit; iin++){
    idx = iin*10;    SK[idx] = 0;  SAK[0][S[0]] = idx;
    S[0] += 1;  DK[idx] = 1;  TotMov += Par[0];
    for (int i1 = 0; i1 < UKRef[idx]; i1++){
      Af = AKRef[idx][i1];
      for (int i2 = 0; i2 < UK[Af]; i2++){
	if (AK[Af][i2] == idx){
	  for (int i3 = i2; i3 < UK[Af]; i3++){
	    AK[Af][i3] = AK[Af][i3+1]; 
	    Rate[Af][i3] = Rate[Af][i3+1];
	  }
	  UK[Af] -= 1; break; 
	}
      }
    }    
  }

  ////////////////////////////////////////

  //  cout << "Here " << endl;

  ostringstream SIRname; SIRname << "Data/SIRResult_Mu20_N60K_In10_"; SIRname << testN;  SIRname <<"Alp_"; SIRname << test; SIRname << ".txt";
  ofstream SIRfile (SIRname.str().c_str());

  ostringstream SKname; SKname << "Data/SK_Mu20_N60K_In10_"; SKname << testN;  SKname << "Alp_"; SKname << test; SKname << ".txt";
  ofstream SKfile (SKname.str().c_str());

  /// Run the simulation //////////////////////
  while (on == 1){
    c = c+1; Tot = TotMov + TotInf;
    ra1 = double(rand()/double(RAND_MAX));    dt = - log(ra1)/Tot;    RT = RT + dt;
    Csum = 0; 
    ra1 = double(rand()/double(RAND_MAX));
    /// Here we move infected between states
    AC = 0; 
    if (TotMov/Tot > ra1){
      for (int i1 = 0; i1 < Nstates-1; i1++){
        for (int i2 = 0; i2 < S[i1]; i2++){
	  Csum += Par[i1]/Tot;
	  if (Csum > ra1){
	    idx = SAK[i1][i2];
	    cin1 = i1; cjn1 = i2;
	    AC = 1;
	    break;
	  }
	}
	if (AC == 1){break;}
      }
     
      /// We have chosen idx to move -> here we move it
      SAK[cin1+1][S[cin1+1]] = idx;
      for (int j = cjn1; j < S[cin1]; j++){SAK[cin1][j] = SAK[cin1][j+1]; }
      SK[idx] += 1;
      S[cin1] -= 1; 
      S[cin1+1] += 1;      
      TotMov -= Par[cin1]; 
      TotMov += Par[cin1+1];     

      /// If it has moven to infectious state we update rates
      if (SK[idx] == Ninfectious){ /// Moves TO infectious State from non-infectious
	for (int i1 = 0; i1 < UK[idx]; i1++){ /// Loop over row idx	  
	  if (SK[AK[idx][i1]] < 0){
	    TotInf += Rate[idx][i1];
	  }
	}
      }
      
      if (SK[idx] == Nstates-1){ /// If this moves to Recovered state
	for (int i1 = 0; i1 < UK[idx]; i1++){ /// Loop over row idx
	  TotInf -= Rate[idx][i1]; }	
	DK[idx] = 2;
	NRecov += 1;
      }
    }
    /// Here we infect new states
    else{
      Csum = TotMov/Tot;
      for (int i1 = Ninfectious; i1 < Nstates-1; i1++){
	for (int i2 = 0; i2 < S[i1]; i2++){
	  idy = SAK[i1][i2];
	  for (int i3 = 0; i3 < UK[idy]; i3++){ 
	    Csum += Rate[idy][i3]/Tot;
	    if (Csum > ra1){
	      idx = AK[idy][i3];	      SK[idx] = 0;
	      SAK[0][S[0]] = idx;	      S[0] += 1;
	      TotMov += Par[0];	      AC = 1;
	      break;
	    }
	  }
	  if (AC == 1){break;}
	}
	if (AC == 1){break;}
      }
    
      /// Here we update infection lists      
      for (int i1 = 0; i1 < UKRef[idx]; i1++){
	acc =0;
	Af = AKRef[idx][i1];
	for (int i2 = 0; i2 < UK[Af]; i2++){
	  if (AK[Af][i2] == idx){
	    if (SK[Af] >= Ninfectious && SK[Af] < Nstates-1){	      
	      TotInf -= Rate[Af][i2];
	    }
	    for (int i3 = i2; i3 < UK[Af]; i3++){
	      AK[Af][i3] = AK[Af][i3+1];
	      Rate[Af][i3] = Rate[Af][i3+1];
	    }
	    UK[Af] -= 1; break;
	  }
	}
      }
    }


    /*
    double ssum = 0; exM = 0;
    for (int i1 = 0; i1 < Nstates-1; i1++){
      exM += S[i1]*Par[i1];
      if (i1 >= Ninfectious){
	for (int i2 = 0; i2 < S[i1]; i2++){
	  ssum += UK[SAK[i1][i2]];
	}
      }
    }
    */
    //    cout << "Check " << ssum << " " << TotInf << " " << " totM " << TotMov << " " << c << endl;

    //    cout << TotInf << " " << ssum << endl;
    
    //    cout << "listing " << c << " TM: " << TotMov << " TI: " << TotInf << " " << cinf << " SK " << SK[0] << " dt " << dt << endl;
    /*
    exM = 0; exI = 0;
    for (int i = 0; i < 12; i++){      exM += S[i]*0.2;    }
      for (int j = -1; j < 1000; j++){
	for (int i = 0; i<13; i++){ 
	  if (j < 0){
	    cout << i << "\t";	  
	  }
	  else if (j < S[i]){
	    if (SK[SAK[i][j]] > 3){
	      exI += UK[SAK[i][j]];
	      if (j < 10){
	      cout << SAK[i][j] << "*" << "\t";
	      }
	    }
	    else{
	      if (j < 10){
	      cout << SAK[i][j] << "\t";
	      }
	    }

	  }
	  else{
	    if (j < 10){
	    cout << "-" << "\t";
	    }
	  }
	}
	if (j<10){
	cout << endl;
	}
      }
      //  cout << " exM " << exM << " exI  " << exI << endl;
      */
    ////////////////////////////////////////////////
    if (nts*click < RT){
      click++; cout << RT << " " << S[0] << " " << S[12] << endl;
    }
      
    /// Criteria to stop
    //    if (ssum < TotInf){ on = 0; cout << " Higher rates than expected " << endl;}
    //    if (ssum > TotInf){ on = 0; cout << " Not all rates added " << ssum << " " << TotInf << " " << c << endl; for (int i = 0; i < 13; i++){cout << S[i] << endl;}}
    if (exM > TotMov+0.1){on = 0; cout << "Move problem " << endl;}
    if (c > 10000000){ on = 0; }
    if (TotInf + TotMov < 0.0001 && TotMov + TotInf > -0.00001){ on = 0; cout << "Equilibrium " << endl;}
    if (S[12] > N0-10){
      cout << "2/3 through " << endl; 
      on = 0;
    }
    
    /// Check for bugs
    if (AC == 0){ cout << "No Chosen rate " << Csum << " " << c << endl; on = 0;}
    if (TotMov < 0 && TotMov > -0.001){TotMov = 0; }
    if (TotInf < 0 && TotInf > -0.001){TotInf = 0; }
    if (TotMov < 0 || TotInf < 0){ cout << "Negative Problem " << " " << TotMov << " " << TotInf << endl; on = 0;  }

  }
  clock_t end1 = clock();  double elapseTime = double(end1 - begin1) / CLOCKS_PER_SEC; cout << elapseTime << endl;
    }
  }
}
  

