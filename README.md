# CEPARCO-Integrating-project-GRP2

Google Colab: https://colab.research.google.com/drive/1i5dJi1BuCl-6fsyGmwBvvDvSCtzewb6k?usp=sharing
Dataset: https://drive.google.com/drive/folders/1qaY8yQJHypNpmHhxHl3clelSWTlreXOG?usp=sharing

# MID REPORT MILESTONE
## Update
- We built a C++ baseline and a CUDA version that rank all 12,953 guesses by entropy for the first turn.
-  Implemented the pattern check, a 243-bin shared-memory histogram per guess (atomicAdd), and a block reduction for the entropy on the GPU.
-  The word lists have 12,953 guesses and 2,309 secrets, not 12,972 and 2,315 as in the proposal.
-  Addressing feedback on repeated letters: green first, then yellow only for unused copies
-  Addressing feedback on deterministic selection: alphabetical order and word replaces best if better by more
  then 0.000001. tie always occurs when the alphabetically first word.
-  Addressing feedback on candidate-set edge cases: at turn 1, the top 10 entropy is displayed, so in turn 2 will be when user input will be required for a guess, this will be planned for final report with the filtering edge cases.

## Results
- CPU time min, max, avg over 30 runs. Top 10 first guesses ranked by entropy
<img width="683" height="222" alt="image" src="https://github.com/user-attachments/assets/7732d9de-0d1e-4b4a-9f8e-d43c8507654f" />


- GPU Histogram Error count and Entropy error count, and Top 10 first guesses
<img width="327" height="297" alt="image" src="https://github.com/user-attachments/assets/ff678673-2794-4b10-abbc-b384c88c3a36" />


- NSYS PROFILE
<img width="1331" height="610" alt="image" src="https://github.com/user-attachments/assets/ef1b21a6-216f-403c-ba66-45f2c3016916" />


- The CPU redoes the same work and compares it with what the GPU gave, histogram counts are whole numbers, so they must match exactly, and entropies can differ in the last digits, so we allow an absolute tolerance of 0.0001 and a relative tolerance of 0.00001. The results show for the GPU, 0 histogram errors and entropy errors, and same top 10 as with the CPU.
- CPU on 30 runs had a min of 1382.51 ms, max of 1821.898 ms, and an average of 1485.4661 ms.
- GPU kernel on 30 runs had a min of  4.262 ms, max of 4.826 ms, and an average of 4.526 ms.

- Speedups:
<img width="792" height="177" alt="image" src="https://github.com/user-attachments/assets/3632594e-77d0-4b47-9445-3e2557659bd3" />

## Plans for final Report
- Complete solver loop that has user guesses
- Add candidate filtering
- Add CPU fallback at a certain threshold from measured turns. Relevant when candidates are already a small pool of choices

# AI Declaration

# Video pitch link
https://drive.google.com/file/d/1U2MF64K19MUbimjEvoSpS-VywXhaIDcU/view?usp=sharing


# Revision
Topic switched from image Gaussian Blur to Wordle Information Entropy Solver

## 0.) Abstract of your proposed project

This project implements a high-performance Wordle solver by parallelizing Shannon entropy calculations. Existing sequential CPU implementations C++ often face performance bottlenecks due to nested loops iterating through large word lists. Our project utilizes SIMT architecture via CUDA to evaluate a guess and find word combinations across thousands of warps. We assign a thread block to each guess, where threads evaluate feedback combinations and tally results into a 243-bin (3^5) histogram using shared memory and operations. A parallel reduction will compute the Shannon entropy to determine the guess with the information gained. The CPU handles sequential operations, including word list loading, user input processing, and memory coordination between the host and device. Using Google Colab’s NVIDIA Tesla T4 cloud GPUs, we will develop a system in C++ and utilize Github for version control. The outputs are a ranked list of optimal next guesses, a benchmark comparing CPU and CUDA execution time across roughly 30 million evaluations, and verification logs confirming that GPU entropy matches the CPU baseline within an absolute tolerance of 1x10^-4 and a relative tolerance of 1x10^-5. 


## 1.) Description of your proposed project

## a. What are the inputs to your project? 

- Secret Word List (Ns=2,315): Contiguous array of 5-letter string arrays representing valid hidden answers.

- Guess Word List (Ng=12,972): Contiguous array of all valid 5-letter words allowed as guesses.

- Game State Constraints: Current feedback bitmasks (Green, Yellow, Gray) supplied dynamically by the user after each turn to filter the remaining active search space. In the game, green means the secret word has this letter and is in the right position, yellow means the letter is in the secret word but not in the right position, and gray means the letter is not inside the secret word.


## b. Propose process

## i. What processes will be parallelized using SIMT (GPU/CUDA) or SIMD-based AVX/AVX2/AVX512 assembly project?

- Combinatorial Pattern Evaluation: Every thread evaluates a distinct (guess, secret_word) pair, simulating Wordle's feedback rules via fast bitwise operations.

- Bucket Frequency Distribution: Tallying the frequency of each resulting feedback pattern across the secret word space.

- Shannon Entropy Calculation & Parallel Reduction: Computing across blocks to identify the guess that maximizes information gain.

<img width="216" height="45" alt="image" src="https://github.com/user-attachments/assets/3e409861-d05e-4607-a578-65de3c0839a9" />


## ii. What existing implementations have been done with the proposed process (i.e., related works)? How would it differ from your proposed implementation if it has been parallelized? 

Related Work: Sequential  C++ information-entropy solvers which iterate through loops on the CPU. The information on the theoretical Wordle solver is presented by 3Blue1Brown (2022), which shows the scores of each guess using the Shannon entropy of its feedback distribution. This was implemented using sequential Python code on the CPU. There are other open source solvers with the same approach. The GPU implementation was by mrowan137 and was written in C++ and HIP. This was used with an AMD GPU. Their implementation ranks guesses using the average number of solution words eliminated, evaluated over triplets, so it did not use Shannon entropy. 

Differentiation: The project differs from the ranked guesses using Shannon entropy with the 243 feedback pattern. The target platform will be performed with NVIDIA CUDA on Google Colab. It will also use a parallel design that uses one thread block per guess, a 243-bin shared-memory histogram built with atomic additions, block-level entropy reduction, and CPU fallback when there are a few guesses left. 




## iii. Discuss plans on how to parallelize your proposed process.

We will use a fixed 243-bin (3^5) histogram per guess to count the possible feedback patterns. We will assign one thread block to each guess, with the threads in the block distributing the evaluations across the remaining secret words. Each thread will evaluate a (guess, secret_word) pair and encode the resulting green, yellow, and gray feedback as one of the 243 patterns. The block will accumulate pattern frequencies in a histogram held in shared memory, using atomic operations, and then reduce over the 243 bins to compute the Shannon entropy of the guess. Then, a separate parallel reduction will compare the entropy scores across all guesses to identify the guess with the highest information gain. We will also explore shared memory, atomic operations, and parallel reduction techniques to improve performance compared to the sequential CPU implementation.


## iv. What other processes are in the project but will not be parallelized.

The loading of the secret_word and guess lists, accepting user input for guesses and feedback, and updating the list of possible secret words based on the feedback received will remain on the CPU. The CPU will also handle displaying the recommended guess and its entropy score, memory coordination, and data transfers between it and the GPU.

## c. What are the outputs of your project?

The output of the project will be the ranked list of optimal next guesses calculated by the highest Shannon entropy score. The main comparison benchmark for comparing sequential CPU performance vs parallel CUDA performance in execution time, GPU kernel time, host-device transfer time, single-threaded C++ time, and speedup for CPU over GPU. The verification logs for the GPU entropy compared to the CPU baseline. It will consist of verifications for the feedback patterns, histogram counts, and selected guesses. The floating-point entropy values will also be compared with an absolute tolerance of 1x10^-4 and a relative tolerance of 1x10^-5 will also be shown.

## 2.) Proposed target implementation platform of the proposed project

The target implementation platform for the project consists of using Google Colab’s NVIDIA Tesla T4 cloud GPU with CUDA support. The core scripts will be implemented using C++. This includes the C++ baseline implementation comparison, the C++ implementation will have the same word list and packed word format as the CUDA implementation. GitHub will be used for version control and documentation of the project.


## 3.) Mid-project milestones of the proposed project

The milestone at week 10 for the mid-project demonstration will feature the functional sequential CPU baseline that has the entropy calculation for a single turn and the basic CUDA kernel executing pattern matching and memory transfer for the host and device of the subset of words. The program should have matching histograms from the baseline, and entropies match within the absolute and relative tolerance. The final delivery at week 13 will be the full integrated CUDA entropy solver code. Benchmark reports for CPU vs GPU execution speeds across the 12,932 x 2,315 = 30,030,180  evaluations. There should be zero integer mismatch, and entropy within tolerance. The GitHub will also be provided, containing the project, documentation, and results.

## 4. References

[1] Mrabet, A. (2023, November 4). Solving Wordle with computer science. Ahmed Mrabet — Data Scientist & ML Engineer. https://ahmedmrabet.me/blog/solving-wordle-with-computer-science 

[2] mrowan137. (n.d.). Wordle-GPU [Source code]. GitHub. https://github.com/mrowan137/wordle-gpu

[3] Sanderson, G. (3Blue1Brown). (2022). Solving Wordle using information theory. Video and source code: 
https://github.com/3b1b/videos/tree/master/_2022/wordle 
