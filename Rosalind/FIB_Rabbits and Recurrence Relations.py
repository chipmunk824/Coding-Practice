def rabbit_pairs(n, k):
    # Base cases: F1 = 1, F2 = 1
    if n == 1 or n == 2:
        return 1
    # Initialize an array to store the number of rabbit pairs
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 1
    # Fill the array using the recurrence relation
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + k * dp[i - 2]
    # Return the result for nth month
    return dp[n]

# Example usage
n = 28  # Number of months
k = 4  # Number of offspring per pair
print(rabbit_pairs(n, k))
