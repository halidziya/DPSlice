#include <iostream>
#include "FastMat.h"
#include <string>
int MAX_SWEEP = 500;
int NINITIAL = 10;
int MAXCOMP = 20;
int BURNIN = 20;
int STEP = (MAX_SWEEP - BURNIN) / 10; // Default value is 10 sample + 1 post burnin

class Restaurant : public Task
{
public:
	Matrix& x;
	vector<Normal> mvns;
	Vector beta;
	Vector u; 
	Vector v;
	Vector labels;
	atomic<int> taskid;
	int nchunks;
	Restaurant(Matrix& data,int nchuks) : x(data), nchunks(nchuks){
		labels = Vector(n);
	}

	void reset(vector<Normal>& mvns, Vector& beta, Vector& u) {
		this->beta = beta;
		this->u = u;
		this->mvns = mvns;
		taskid = 0;
	}

	void run(int id) {
		SETUP_ID()
		int taskid = this->taskid++; // Oth thread is the main process
		auto range = trange(n, nchunks, taskid); // 2xNumber of Threads chunks
		//cout << range[1] << endl;
		int NTABLE = mvns.size();
		Vector likelihoods(NTABLE);
		for (auto i=range[0];i<range[1];i++) // Operates on its own chunk
		{

			for (auto j = 0;j < NTABLE;j++)
			{
				if (beta[j] > u[i])
					likelihoods[j] = mvns[j].likelihood(x(i)); //**
				else
					likelihoods[j] = -INFINITY;
			}
			labels[i] = sampleFromLog(likelihoods); //**
		}
	}

};


class Collector : public Task
{
public:
	Matrix& x;
	Vector& labels;
	Vector count;
	vector<Matrix> scatter;
	vector<Vector> sum;
	atomic<int> taskid;

	Collector(Matrix& x, Vector& labels) : x(x), labels(labels) {
	}

	void reset() {
		int vsize = labels.maximum()+1;
		count = zeros(vsize);
		scatter = vector<Matrix>(vsize,zeros(d,d));
		sum = vector<Vector>(vsize,zeros(d));
		taskid = 0;
	}

	void run(int id) {
		// Use thread specific buffer
		SETUP_ID()
		int taskid = this->taskid++;
		int maxlab = labels.maximum();
		int label;
		for (int i = 0;i < n;i++)
		{
			label = labels(i);
			if (label == taskid) // Distributed according to label id
			{
				count[label] += 1;
				sum[label] += x(i);
				scatter[label] += x(i)>>x(i); // Second moment actually
			}
		}
	}
};



int relabel(Vector& labels)
{
	Vector ulabels = labels.unique();
	Vector dict(round(labels.maximum()) + 1);
	int j = 0;
	for (int i = 0;i < ulabels.n;i++)
		dict[ulabels(i)] = j++;
	for (int i = 0;i<labels.n;i++)
		labels[i] = dict[labels(i)];
	return ulabels.n;
}

Vector stickBreaker(double ustar, double betastar = 1.0,double alpha = 1)
{
	Dirichlet  ds(v({ 1, alpha })); // Beta
	Vector lengths(MAXCOMP);
	for (int i = 0;i < MAXCOMP;i++)
		lengths[i] = ds.rnd()[0];
	Vector beta = zeros(MAXCOMP + 1);
	int i = 0;
	double totallength = betastar;
	double betasum=0;
	for (i=0;i<MAXCOMP;i++)
		{
		beta[i] = betastar*lengths[i];
		betasum += beta[i];
		betastar = betastar*(1 - lengths[i]);
		if (betastar < ustar)
			break;
		}
	beta.resize(i+2);
	beta[i + 1] = totallength - betasum;
	return beta;
}

double gammalnd(int x) // Actually works on x/2 
{
	double res = 0;
	for (auto i = 0; i < d; i++)
	{
		res += gl_pc[x - i];
	}
	return (log(M_PI)*d*(d - 1) / 4) + res;
}

double logpi = log(M_PI);

Matrix SliceSampler(Matrix& x, double m, double kappa, double gamma, Vector& mu0, Matrix& Psi, ThreadPool& workers,Vector& likelihoods,Vector initialLabels = v({}),int empricalCovariance=0)
{
	// INITIALIZATION
	// Point level variables
	int NTABLE = NINITIAL;
	Vector u = zeros(n);
	Vector beta = ones(NINITIAL);
	beta /= NINITIAL;

	// Table level variables
	vector<Normal> mvns(NINITIAL,Normal(d));

	Restaurant r(x,nthd*2); //2*nthd chunks
	if (initialLabels.n==0)
		r.labels = rand(n, NTABLE);
	else
		r.labels = initialLabels;
	Collector  c(x,r.labels);
	IWishart priorcov(Psi,m); // No need to create in every iteration
	int nlabelsample = ((MAX_SWEEP - BURNIN) / STEP);
	Matrix sampledLabels(nlabelsample, n);
	//SIMULATION
	for (auto iter = 0; iter < MAX_SWEEP; iter++)
	{
		if (iter % 10 == 0)
			cout << "Iter : " <<  iter << endl;
		//Collect Statistics
		c.reset();
		for (auto i = 0; i < NTABLE; i++) { // Each label collects its statistics async.
			workers.submit(c);
		}
		workers.waitAll();

		double totallikelihood = 0;
		for (int i = 0;i < c.count.n;i++) // Jchang's formula for joint marginal distribution
		{
			int  n = c.count[i];
			double s1 = kappa + n;
			Vector& s = c.sum[i];
			Vector& diff = (mu0 - (s / n));
			Matrix& ss = Psi + c.scatter[i] - ((c.sum[i] >> c.sum[i]) / n) + (diff >> diff)*(kappa*n / (kappa + n));
			ss = ss / (n + m);
			totallikelihood += -0.5*n*d*logpi - gammalnd(m) + gammalnd(n + m) - (0.5*(n+ m))*(d*log(n + m)
				+ 2 * ss.chol().sumlogdiag()) + (0.5*m)*(d*log(m) + 2 * (Psi / m).chol().sumlogdiag()) - 0.5*d*log((n + kappa) / kappa) + log(gamma) + gl_pc[n * 2];
		}
		likelihoods[iter] = totallikelihood + gl_pc[gamma * 2] - gl_pc[(gamma + n) * 2];


	    // Create Betas
		Vector alpha = c.count.append(gamma);
		Dirichlet dr(alpha);
		beta = dr.rnd();
		// Sample U
		u = rand(n);
		u *= beta[r.labels];
		
		//New Sticks
		Vector newsticks = stickBreaker(u.minimum(), beta[beta.n - 1]);
		beta.resize(beta.n-1);
		beta = beta.append(newsticks);
		NTABLE = beta.n;

		// Sample from Parameter Posterior or From Prior
		mvns.resize(NTABLE,Normal(d));
		for (int i = 0;i < NTABLE;i++)
		{
			if (i < c.count.n) // Used tables
			{
				int n = c.count[i];
				Vector meandiff = ((c.sum[i] / n)-mu0);
				IWishart posteriorcov(Psi + c.scatter[i] - ((c.sum[i]>>c.sum[i])/n) + (meandiff>>meandiff)*(kappa*n / (kappa + n)), m + n);
				Matrix sigma = posteriorcov.rnd();
				Normal posteriormean(( mu0*kappa + c.sum[i])/n, sigma / (kappa+n));
				mvns[i] = Normal(posteriormean.rnd(),sigma);
			}
			else  // Empty Tables , Sample from Prior
			{
				Matrix sigma = priorcov.rnd();
				mvns[i].cholsigma = sigma.chol();
				Normal priormean(mu0, sigma / kappa);
				mvns[i].mu =  priormean.rnd();
			}
		}

		if (empricalCovariance == 1) // Updates the PSI based on average covariance, emprical default false
		{
			Psi = zeros(d, d);
			for (int i = 0;i < c.count.n;i++)
				Psi += (c.scatter[i] - ((c.sum[i] >> c.sum[i]) / c.count[i]));
			Psi /= (n/(m-d-1));
			Psi.print();
		}

		//Sample labels
		r.reset(mvns, beta, u);
		for (auto i = 0; i < r.nchunks; i++) {
			workers.submit(r);
		}
		workers.waitAll();

		NTABLE = relabel(r.labels); // Get unique ones , remove ones with 0 prob


		if (iter >= BURNIN && (iter - BURNIN) % STEP == 0) {
			int li = (((iter - BURNIN) / STEP));
			for (auto i = 0; i < n;i++)
				sampledLabels(li)[i] = r.labels[i];
		}
		
	}

	return sampledLabels;
}


int main(int argc, char** argv)
{

	Matrix x;
	Matrix psi;
	Vector mu0;
	Vector hyperparams;
	Vector initialLabels = v({});
	generator.seed(time(NULL));
	srand(time(NULL));
	if (argc > 1)
		x.readBin(argv[1]);

	cout << argv[1] << endl;
	cout << "NPOINTS :" << x.r << " NDIMS:" << x.r << endl;
	nthd = thread::hardware_concurrency();
	n = x.r; // Number of Points
	d = x.m;
	init_buffer(nthd, x.m);
	cout << " Available number of threads : " << nthd << endl;
	precomputeGammaLn(2 * n + 100 * d);


	// Hyper-parameters with default values
	if (x.data == NULL)
	{
		cout << "Usage: " << "dpsl.exe datafile.matrix [hypermean.matrix] [hyperscatter.matrix] [params.matrix (d,m,kappa,gamma)] [initiallabels.matrix]";
		return -1;
	}
	if (argc > 2)
		mu0.readBin(argv[2]);
	else
		mu0 = x.mean();
	if (argc > 3)
		psi.readBin(argv[3]);
	else
		psi = (eye(d)*(x.m + 3));

	if (argc > 4)
	{
		hyperparams.readBin(argv[3]);
		m = hyperparams[1];
		kappa = hyperparams[2];
		gamma = hyperparams[3];
	}
	else
	{
		m = x.m + 3;
		kappa = 1;
		gamma = 1;
	}
	if (argc > 5)
		initialLabels.readBin(argv[5]);

	
	ThreadPool tpool(thread::hardware_concurrency());
	debugMode(1);
	step();
	Vector likelihoods(MAX_SWEEP);
	auto labels = SliceSampler(x,m,kappa,gamma,mu0,psi,tpool,likelihoods,initialLabels); // data,m,kappa,gamma,mean,cov 
	string filename = argv[1];
	labels.writeBin(filename.append( ".labels").c_str());
	filename = argv[1];
	likelihoods.writeBin(filename.append(".likelihood").c_str());
	step();

}