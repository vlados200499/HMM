# Hidden Markov Models (HMM) Algorithms

This repository provides implementations and formulas for fundamental algorithms in Hidden Markov Models (HMM). The following algorithms are covered:

- **Forward Algorithm:** Computes the joint probability of an observation sequence and state sequence in an HMM.

- **Viterbi Algorithm:** Finds the most likely state sequence given an observation sequence in an HMM.

- **Baum-Welch Algorithm (Expectation-Maximization):** Estimates the parameters of an HMM from an observed sequence using the EM algorithm.

## Formulas

### Forward Algorithm

#### Initialization:
   \[\alpha_1(i) = \pi_i \cdot B_{i, O_1}, \quad 1 \leq i \leq N\]

#### Recursion:
   \[\alpha_{t+1}(j) = \left( \sum_{i=1}^N \alpha_t(i) \cdot A_{ij} \right) \cdot B_{j, O_{t+1}}, \quad 1 \leq j \leq N, \quad 1 \leq t \leq T-1\]

### Viterbi Algorithm

#### Initialization:
   \[\delta_1(i) = \pi_i \cdot B_{i, O_1}, \quad 1 \leq i \leq N\]
   \[\psi_1(i) = 0, \quad 1 \leq i \leq N\]

#### Recursion:
   \[\delta_{t+1}(j) = \max_{1 \leq i \leq N} (\delta_t(i) \cdot A_{ij}) \cdot B_{j, O_{t+1}}, \quad 1 \leq j \leq N, \quad 1 \leq t \leq T-1\]
   \[\psi_{t+1}(j) = \arg\max_{1 \leq i \leq N} (\delta_t(i) \cdot A_{ij}), \quad 1 \leq j \leq N, \quad 1 \leq t \leq T-1\]

#### Termination:
   \[P^* = \max_{1 \leq i \leq N} \delta_T(i)\]
   \[q_T^* = \arg\max_{1 \leq i \leq N} \delta_T(i)\]

#### Backtracking:
   \[q_t^* = \psi_{t+1}(q_{t+1}^*), \quad T-1 \geq t \geq 1\]

### Baum-Welch Algorithm (Expectation-Maximization)

#### E-Step (Expectation Step):

##### Forward Pass:
   \[\alpha_1(i) = \pi_i \cdot B_{i, O_1}, \quad 1 \leq i \leq N\]
   \[\alpha_{t+1}(j) = \left( \sum_{i=1}^N \alpha_t(i) \cdot A_{ij} \right) \cdot B_{j, O_{t+1}}, \quad 1 \leq j \leq N, \quad 1 \leq t \leq T-1\]

##### Backward Pass:
   \[\beta_T(i) = 1, \quad 1 \leq i \leq N\]
   \[\beta_t(i) = \sum_{j=1}^N A_{ij} \cdot B_{j, O_{t+1}} \cdot \beta_{t+1}(j), \quad 1 \leq i \leq N, \quad T-1 \geq t \geq 1\]

##### Posterior Probabilities:
   \[\gamma_t(i) = \frac{\alpha_t(i) \cdot \beta_t(i)}{\sum_{j=1}^N \alpha_t(j) \cdot \beta_t(j)}, \quad 1 \leq i \leq N, \quad 1 \leq t \leq T\]
   \[\xi_t(i, j) = \frac{\alpha_t(i) \cdot A_{ij} \cdot B_{j, O_{t+1}} \cdot \beta_{t+1}(j)}{\sum_{k=1}^N \sum_{l=1}^N \alpha_t(k) \cdot A_{kl} \cdot B_{l, O_{t+1}} \cdot \beta_{t+1}(l)}, \quad 1 \leq i, j \leq N, \quad 1 \leq t \leq T-1\]

#### M-Step (Maximization Step):

##### Update Initial Probabilities:
   \[\pi_i' = \gamma_1(i), \quad 1 \leq i \leq N\]

##### Update Transition Matrix:
   \[A'_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}, \quad 1 \leq i, j \leq N\]

##### Update Observation Matrix:
   \[B'_{ij} = \frac{\sum_{t=1}^T \delta_{O_t, v_j} \cdot \gamma_t(i)}{\sum_{t=1}^T \gamma_t(i)}, \quad 1 \leq i \leq N, \quad 1 \leq j \leq M\]

**Note:**

- \( \delta_{ij} \) is the Kronecker delta (equals 1 if \(i = j\), and 0 otherwise).
- \(N\) is the number of states in the HMM.
- \(M\) is the number of possible symbols (observable states).
- \(T\) is the length of the observation sequence.
- \(\pi_i\) is the initial probability of state \(s_i\).
- \(A_{ij}\) is the probability of transition from state \(s_i\) to state \(s_j\).
- \(B_{ij}\) is the probability of observing symbol \(v_j\) in state \(s_i\).
- \(O_t\) is the observed symbol at time \(t\).

## Usage
Provide instructions on how to use the code or algorithms in this repository.

## License
This project is licensed under the [MIT License](LICENSE).
