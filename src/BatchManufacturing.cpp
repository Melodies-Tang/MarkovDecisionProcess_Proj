/*
QUESTION
Write programs to solve batch manufacturing problem.
Formulate and solve both discounted and average cost problems.
Realize both value iteration and policy iteration algorithms.
The parameters c, K, n, p, α as inputs.

Experiments and analysis
Run the code with different parameter inputs to show the performance of value iteration and policy iteration.
For discounted problem, consider exercise 7.8.
For average cost problem, write code to find the threshold by stationary analysis.

Compare the optimal policies of the discounted problem with different values of α and the average cost problem.
Submit a report by Dec. 30, 23:59:59, 2021 (include source code in appendix).
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <vector>
#include <algorithm>
#include <D:\mingw64\include\eigen3\Eigen\Dense>
#include <iostream>
#define DISCOUNTED 0
#define AVERAGE 1
#define VAL 0
#define POL 1
#define THR 2
#define STA 3
#define PLV 4
#define FULFILL 1
#define LEAVE 0
#define ACCURACY 1e-4
#define INF 1.0 / 0.0
using namespace std;

// ASSUMPTION: the policy is certain, not stochastic

/*
VALUE ITERATION
* initialize value function.
* loop through the value function (array) until it converges (judged by a threshold of value change).
    each loop: find the optimal expectation of value (through available actions).
* after convergence, traverse the value function to construct the optimal policy (by optimize Expectation(R(current state) + J(next state))).
** Additional: implement plain method for average problem (when alpha=1, use average value function instead.) Slow and cannot converge precisely
*/

vector<int> valueIteration(double c, double K, int n, double p, double alpha) {
    // initialize value function (total cost of state till current epoch (day))
    vector<double> val(n + 1);

    // precalculate of constant coeficients, for acceleration
    const double ORDER = alpha * p;
    const double NORDER = alpha * (1 - p);

    // consider average cost problem, val(i) = g(i) / N
    bool avg = (alpha == 1.0);
    double denom = 2.0;

    // iteration, each loop means transition from stage(k) to stage(k+1)
    while (true) {
        bool terminate = true;

        vector<double> preState(val.begin(), val.end());
        double fulfill = K + NORDER * preState[0] + ORDER * preState[1];  // precalculate
        // Jk+1(i) = min[K + α(1 - p)Jk(0) + αpJk(1), ci + α(1 - p)Jk(i) + αpJk(i + 1)]
        for (int i = 0; i <= n; ++i) {
            double update = fulfill;
            if (i != n) update = min(update, c*i + NORDER * preState[i] + ORDER * preState[i+1]);  // for state(n) only one operation available
            val[i] = update;
            // calculate fining gain (in avg problem: improvement from pov of AVERAGE)
            double change = avg ? abs(val[i]/denom - preState[i]/(denom - 1)) : abs(val[i] - preState[i]);
            if(change > ACCURACY) terminate = false;
        }

        if (terminate) break;
        if (avg) denom += 1.0;
    }
    // calculate optimal policy
    vector<int> policy(n + 1, LEAVE);
    double fulfill = K + NORDER * val[0] + ORDER * val[1];  // precalculate
    int i = 1;
    while (i < n && c*i + NORDER * val[i] + ORDER * val[i+1] <= fulfill) ++i;
    while (i <= n) policy[i++] = FULFILL;
    return policy;
}


/*
POLICY ITERATION (standard)
* initialize value function and origin policy (arbitrarily).
* loop through policies of states, switch policy if better value can be reached.
    before policy comparison, evaluate each state based on current policy, for each loop.
* loop until no policy changed for a whole loop.
* method of eval mentioned above is just like the value iteration, besides the value is determined by given certain policy
    instead of optimal expectation gained by comparison between available policies.
*/

vector<double> updateValue(vector<int>& policy, double c, double K, int n, const double ORDER, const double NORDER) {
    vector<double> val(n + 1);
    while (true) {
        // store value of last iteration
        vector<double> preState(val.begin(), val.end());
        bool terminate = true;
        double fulfill = K + NORDER * preState[0] + ORDER * preState[1];  // precalculate
        for (int i = 0; i <= n; ++i) {
            // calculate value of state under the policy provided (take expectation as the evaluation of state value)
            // may take several iterations to convergence to the actual value function
            double update = fulfill;
            if (policy[i] == LEAVE) {
                update = c*i + NORDER * preState[i] + ORDER * preState[i + 1];
            }
            val[i] = update;
            if (abs(val[i] - preState[i]) > ACCURACY) terminate = false;
        }
        if (terminate) break;
    }
    return val;
}

vector<int> policyIteration(double c, double K, int n, double p, double alpha) {
    // precalculate of constant coeficients, for acceleration
    const double ORDER = alpha * p;
    const double NORDER = alpha * (1 - p);

    // initialize policy for each state
    // according to exe7.8, use a threshold policy, since the optimal will be the same type
    vector<int> policy(n + 1, FULFILL);
    policy[0] = LEAVE;

    // initalize value function, no need for initial value
    vector<double> val(n + 1);

    // iteration, each loop means transition from stage(k) to stage(k+1)
    while (true) {
        bool terminate = true;
        // Why several epochs needed?
        val = updateValue(policy, c, K, n, ORDER, NORDER);
        double fulfill = K + NORDER * val[0] + ORDER * val[1];  // irrelevant to state, precalculate
        for (int i = 1; i < n; ++i) {
            // update dependent on value function derived from last policy
            int pk = policy[i];
            double leave = c*i + NORDER*val[i] + ORDER*val[i+1];
            if (fulfill < leave) policy[i] = FULFILL;
            else policy[i] = LEAVE;
            if (policy[i] != pk) terminate = false;
        }
        if (terminate) break;
    }
    return policy;
}


/*
By Bellman's equation, monotonicity of J* can be proved
Since J∗(i) is nondecreasing in i, ci + α(1 - p)J∗(i) + αpJ∗(i + 1) is increasing in i
There exist a threshold m such that
c(m - 1) + α(1 - p)J∗(m - 1) + αpJ∗(m)   (-> state(m-1), not process, named lower)
≤ K + α(1 - p)J∗(0) + αpJ∗(1)   (-> process)
≤ cm + α(1 - p)J∗(m) + αpJ∗(m + 1)   (-> state(m), not process, named upper)
The optimal threshold policy: to process when i ≥ m, and not to process otherwise
*/

vector<int> thresholdIteration(double c, double K, int n, double p, double alpha) {
    // precalculate of constant coeficients, for acceleration
    const double ORDER = alpha * p;
    const double NORDER = alpha * (1 - p);

    // initialize policy for each state (fulfill unless state(0) initially)
    vector<int> policy(n + 1);
    int m = 1;

    // initalize value function
    vector<double> val(n + 1);

    // iteration, each loop means transition from stage(k) to stage(k+1)
    // note that process cost is irrelevant to specific state, use binary search for each loop
    while (true) {
        // modify policy to be consistent with threshold
        for (int i = 0; i < m; ++i) policy[i] = LEAVE;
        for (int i = m; i <= n; ++i) policy[i] = FULFILL;

        val = updateValue(policy, c, K, n, ORDER, NORDER);

        double fulfill = K + NORDER * val[0] + ORDER * val[1];  // irrelevant to state, precalculate

        // bisearch to get min m that satisfy the equation above
        int l = 1, r = n;
        while (l <= r) {
            int mid = (r - l) / 2 + l;
            if (mid == n) break;
            // double lower = c * mid - 1 + NORDER * val[mid - 1] + ORDER * val[mid];
            double upper = c * mid + NORDER * val[mid] + ORDER * val[mid + 1];
            if (upper <= fulfill) l = mid + 1;
            else r = mid - 1;
        }
        if (l == m) break;  // no improvement (convergence)
        else m = l;
    }
    return policy;
}


/*
AVERAGE COST PROBLEM
* A stationary system with finite number of states and controls.
    For a single recurrent class, the cost incurred up to any fixed time T does not matter.
    The optimal cost is independent of initial state, i.e. J∗(i) = λ∗, ∀i
    minimum average cost = minimum average cost of a cycle = minimum cost of a SSP
* Find a stationary policy µ that minimizes the expected cost per transition within a cycle
    Minimizing Cnn(µ) - Nnn(µ)λ∗ is a SSP problem with per-stage cost g(i,u) - λ∗.
    If µ∗ is optimal, Cnn(µ∗) - Nnn(µ∗)λ∗ = 0 since λ∗ is achievable.
    The optimal average cost λ∗ is the same for all initial states
* Let h∗(i) the optimal cost of this SSP problem when starting at the nontermination states i = 1,...,n.
    h∗(i) = min[g(i, u) - λ∗ + SUM(pij(u)h∗(j))], i = 1,...,n-1   (similar to plain cost function, instead of minusing λ) 
    h∗(n) = Cnn(µ∗) - Nnn(µ∗)λ∗ = 0
* If a scalar λ and a vector h = {h(1),..., h(n)} satisfies Bellman’s equation, then λ is the optimal average cost for all initial states
* For batch manufacturing problem:
    λ∗ + h∗(i) = min[K + (1 - p)h∗(0) + ph∗(1), ci + (1 - p)h∗(i) + ph∗(i + 1)], i = 1,...,n-1
    λ∗ + h∗(n) = K + (1 - p)h∗(0) + ph∗(1)
  the stationary policy µ(i) is optimal if the above minimum for all i.
  In addition, there is a unique vector h∗ with h∗(n) = 0 (initial condition. If not, there are several solutions)
* λµ satisfies the following equation for all i:
    λµ + hµ(i) = g(i, µ(i)) + SUM(pijh(j)), given policy µ, unique when h(n) = 0
*/

/*
* RELATIVE VALUE ITERATION
* As mentioned above, λ can be solved when h given
*/

vector<int> valueIteration(double c, double K, int n, double p) {
    // initalize vector h and scalar lambda, and 'fixed state'
    vector<double> h(n + 1, K);
    double lambda = INF;
    int s = rand() % n;
    // // (page 432, DPOC_v1) τ∈(0,1) variant is required for ensuring convergence
    // // but this approach gives wrong answer (inconsistent with other methods)
    // double tau = 1;

    // iteration, each loop means transition from stage(k) to stage(k+1)
    while (true) {
        bool terminate = true;

        // update lambda based on latest h, using predetermined fixed state s
        double lambdaK = min(K + (1-p)*h[0] + p*h[1], c*s + (1-p)*h[s] + p*h[s+1]);
        // double lambdaK = min(K + tau*(1-p)*h[0] + tau*p*h[1], c*s + tau*(1-p)*h[s] + tau*p*h[s+1]);
        if (abs(lambdaK - lambda) > ACCURACY) terminate = false;
        lambda = lambdaK;

        // update h based on lamda(k)
        vector<double> preH(h.begin(), h.end());
        for (int i = 0; i <= n; ++i) {
            double g = K + (1-p)*preH[0] + p*preH[1];
            if (i != n) g = min(g, c*i + (1-p)*preH[i] + p*preH[i+1]);
            // double g = min(K + tau*(1-p)*preH[0] + tau*p*preH[1],
            //                (i == n ? INF : c*i + tau*(1-p)*preH[i] + tau*p*preH[i+1]));
            h[i] = g - lambda;
            if (abs(h[i] - preH[i]) > ACCURACY) terminate = false;
        }

        if (terminate) break;
    }

    // calculate optimal policy through final relative cost h
    vector<int> policy(n + 1, LEAVE);
    double fulfill = K + (1-p)*h[0] + p*h[1];  // precalculate
    int i = 1;  // for state(0), it's not allowed to manufacture
    while (i < n && c*i + (1-p)*h[i] + p*h[i+1] <= fulfill) ++i;
    while (i <= n) policy[i++] = FULFILL;
    return policy;
}


/*
STANDARD POLICY ITERATION
* At iteration k, we have a stationary µk
* Policy iteration: Solving linear equations (e):
    λk + hk(i) = g(i, µk(i)) + SUM(pij(µk(i))hk(j))
    hk(n) = 0
* Policy improvement:
    µk+1(i) = argmin[g(i, u) + SUM(pij(u)hk(j))]
*/

pair<vector<double>, double> updateH(vector<int>& policy, double c, double K, int n, double p) {
    vector<double> h(n + 1);
    double lambda;
    // Solve the following equation (current method: brute-force)
    // λk + hk(i) = g(i, µk(i)) + SUM(pij(µk(i))hk(j))    (1)
    // hk(n) = 0    (2)
    while (true) {
        bool term = true;
        vector<double> preH(h.begin(), h.end());
        lambda = K + (1-p)*h[0] + p*h[1];
        // keep hn=0 to take condition for unique solution into consideration
        for (int i = n - 1; i >= 0; --i) {
            if (policy[i] == FULFILL) h[i] = K + (1-p)*preH[0] + p*preH[1] - lambda;
            else h[i] = c*i + (1-p)*preH[i] + p*preH[i+1] - lambda;
            if (abs(h[i] - preH[i]) > ACCURACY) term = false;
        }
        if (term) break;
    }
    return {h, lambda};
}

vector<int> policyIteration(double c, double K, int n, double p) {
    vector<double> preH(n + 1);
    double preLambda = INF;

    // initialize policy
    // according to exe7.8, use a threshold policy, since the optimal will be the same type
    vector<int> policy(n + 1, FULFILL);
    policy[0] = LEAVE;

    while (true) {
        bool terminate = true;

        // update h and lambda by solving the linear equation (e)
        // auto [h, lambda] = updateH(policy, c, K, n, p);
        pair<vector<double>, double> hl = updateH(policy, c, K, n, p);
        vector<double>& h = hl.first;
        double lambda = hl.second;
        if (abs(lambda - preLambda) > ACCURACY) terminate = false;
        else {
            for (int i = 0; i <= n; ++i) {
                if (abs(preH[i] - h[i]) > ACCURACY) {terminate = false; break;}
            }
        }
        preH = h;
        preLambda = lambda;
        
        // do policy improvement based on latest lambda and h
        double fulfill = K + (1-p)*h[0] + p*h[1];
        for (int i = 1; i < n; ++i) {
            int improve = c*i + (1-p)*h[i] + p*h[i+1] < fulfill ? LEAVE : FULFILL;
            if (improve != policy[i]) {
                policy[i] = improve;
                terminate = false;
            }
        }

        if (terminate) break;
    }
    return policy;
}


/*
STATIONARY ANALYSIS
Given threshold m, a markov chain of state(0) to state(m) can be determined
The transition matrix P of this class is available, then π (π = π[P])
Average cost (also expectation) f(m) = SUM(c*i*πi) + K*πm
Iterate m to minimize f(m)
*/

vector<int> stationaryAnalyse(double c, double K, int n, double p) {
    double minf = INF;
    int optimal_m = 1;
    
    for (int m = 1; m <= n; ++m) {
        // initialize transition matrix P
        Eigen::MatrixXf P = Eigen::ArrayXXf::Zero(m + 1, m + 2);
        for (int i = 0; i < m; ++i) {
            P(i, i) = 1 - p;
            P(i, i + 1) = p;
        }
        P(m, 0) = 1 - p;
        P(m, 1) = p;
        // cout << "P:\n" << P << endl;

        // construct coefficient matrix A and b
        Eigen::MatrixXf A = P.transpose() - Eigen::MatrixXf::Identity(m + 2, m + 1);
        for (int i = 0; i <= m; ++i) A(m + 1, i) = 1;
        // cout << "A:\n" << A << endl;
        Eigen::VectorXf b = Eigen::ArrayXf::Zero(m + 2);
        b(m + 1) = 1;
        // cout << "b:\n " << b << endl;

        // solve π = πP (after transpose), condition SUM(π)=0
        Eigen::VectorXf state = A.colPivHouseholderQr().solve(b);
        // cout << "π:\n" << state << endl;

        Eigen::VectorXf mul = Eigen::ArrayXf::Zero(m + 1);
        for (int i = 1; i < m; ++i) mul(i) = c * i;
        mul(m) = K;
        double fm = mul.dot(state);
        if (fm < minf) {
            optimal_m = m;
            minf = fm;
        }
    }
    vector<int> policy(n + 1, LEAVE);
    for (int i = optimal_m; i <= n; ++i) policy[i] = FULFILL;
    return policy;
}

double getAndShowPolicy(int problem, int method, double c, double K, int n, double p, double alpha) {
    vector<int> policy(n + 1);
    clock_t startTime, endTime;
    startTime = clock();
    if (problem == DISCOUNTED) {
        switch (method) {
        case VAL:
            policy = valueIteration(c, K, n, p, alpha);
            printf("Value iteration      : ");
            break;
        case POL:
            policy = policyIteration(c, K, n, p, alpha);
            printf("Strategy (normal)    : ");
            break;
        case THR:
            policy = thresholdIteration(c, K, n, p, alpha);
            printf("Strategy (threshold) : ");
            break;
        }
    } else if (problem == AVERAGE) {
        switch (method) {
        case PLV:
            policy = valueIteration(c, K, n, p, alpha);
            printf("Value iteration (Jk) : ");
            break;
        case VAL:
            policy = valueIteration(c, K, n, p);
            printf("Value iteration (Hk) : ");
            break;
        case POL:
            policy = policyIteration(c, K, n, p);
            printf("Strategy iteration   : ");
            break;
        case STA:
            policy = stationaryAnalyse(c, K, n, p);
            printf("Stationay analysis   : ");
            break;
        }
    }
    endTime = clock();
    double usage = (double)1000 * (endTime - startTime) / CLOCKS_PER_SEC;
    int m = 0;
    while (policy[m] == LEAVE) ++m;
    printf("Solve threshold: %d ", m);
    // printf("[ ");
    // for (int i = 0; i < n; ++i) printf("%d, ", policy[i]);
    // printf("%d ]", policy[n]);
    printf("in %.5f ms", usage);
    printf("\n");
}

int main() { 
    // special: value iteration with alpha implemented the problem with alpha == 1
    // n (int): max number of orders (when arrived at, must manufacture), n >= 1
    // c (double): hold cost for every order not fulfilled yet
    // K (double): cost of manufacturing (irrelevant to num_orders)
    // p (double): probability that got one new order (no order otherwise)
    // α (double): discount factor, 0 <= alpha < 1 for discounted problem, =1 only for val-average method
    // vector<double> c{1, 3, 8, 5, 0, 5};
    // vector<double> K{5, 5, 10, 18, 1, 0.5};
    // vector<int> n{10, 10, 20, 20, 10, 10};
    // vector<double> p{0.5, 0.5, 0.7, 0.7, 0.6, 0.65};
    // vector<double> alpha{0.9, 0.9, 0.9, 0.9, 0.95, 0.85};

    srand(41561256);
    int num_tests = 50;
    vector<double> c(num_tests);
    vector<double> K(num_tests);
    vector<int> n(num_tests);
    vector<double> p(num_tests);
    vector<double> alpha(num_tests);
    for (int i = 0; i < num_tests; ++i) {
        c[i] = rand() % 20 * (rand() * 3 / (double)RAND_MAX);
        K[i] = (rand() % 3 + rand() / (double)RAND_MAX) * c[i];
        n[i] = rand() % 20 * (rand() * 3 / (double)RAND_MAX) + 1;
        p[i] = rand() / (double)RAND_MAX;
        alpha[i] = rand() / (double)RAND_MAX;
    }
    
    for (int i = 0; i < num_tests; ++i) {
        printf("Discounted problem with: c=%.3f, K=%.3f, n=%d, p=%.3f, α=%.3f\n", c[i], K[i], n[i], p[i], alpha[i]);
        getAndShowPolicy(DISCOUNTED, VAL, c[i], K[i], n[i], p[i], alpha[i]);
        getAndShowPolicy(DISCOUNTED, POL, c[i], K[i], n[i], p[i], alpha[i]);
        getAndShowPolicy(DISCOUNTED, THR, c[i], K[i], n[i], p[i], alpha[i]);
        printf("\n");
    }

    for (int i = 0; i < num_tests; ++i) {
        printf("Average cost problem with: c=%.3f, K=%.3f, n=%d, p=%.3f\n", c[i], K[i], n[i], p[i]);
        // the first two maybe wrong
        getAndShowPolicy(AVERAGE, PLV, c[i], K[i], n[i], p[i], 1.0);
        getAndShowPolicy(AVERAGE, VAL, c[i], K[i], n[i], p[i], 1.0);
        getAndShowPolicy(AVERAGE, POL, c[i], K[i], n[i], p[i], 1.0);
        getAndShowPolicy(AVERAGE, STA, c[i], K[i], n[i], p[i], 1.0);
        printf("\n");
    }

    return 0;
}
