def find_largest_common_substring(s1, s2):

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    lcs_end = 0  # Store the end index of the longest common substring in s1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    lcs_end = i
            else:
                dp[i][j] = 0

    return s1[lcs_end - longest : lcs_end] if longest > 0 else "", longest


def commonality(s1, s2):
    total_common_length = 0
    original_length = len(s1) + len(s2)

    while True:
        common_substring, length = find_largest_common_substring(s1, s2)
        # print(common_substring)
        if not common_substring:
            break
        total_common_length += length

        s1 = s1.replace(common_substring, "", 1)
        s2 = s2.replace(common_substring, "", 1)

    # Calculate the commonality metric
    if original_length == 0:
        return 0
    # print(2 * (total_common_length / original_length))
    return 2 * (total_common_length / original_length)


def commonality_and_unmatched(s1, s2):
    total_common_length = 0
    original_length_s1 = len(s1)
    original_length_s2 = len(s2)

    while True:
        common_substring, length = find_largest_common_substring(s1, s2)
        if not common_substring:
            break
        total_common_length += length

        # Remove the common substring from both s1 and s2
        s1 = s1.replace(common_substring, "", 1)
        s2 = s2.replace(common_substring, "", 1)

    # Return the total common length and the unmatched lengths
    uLens1 = len(s1)
    uLens2 = len(s2)

    return total_common_length, uLens1, uLens2


def difference(s1, s2, p=0.6):
    _, uLens1, uLens2 = commonality_and_unmatched(s1, s2)

    numerator = uLens1 * uLens2
    denominator = p + (1 - p) * (uLens1 + uLens2 - uLens1 * uLens2)

    if denominator == 0:
        return 0
    # print(numerator / denominator)
    return numerator / denominator


def winkler(s1, s2, comm_value, diff_value, l):

    # _, common_length = find_largest_common_substring(s1, s2)
    prefix = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            break
        prefix += 1

    winkler_value = (1 - (comm_value - diff_value)) * min(prefix, 4) * l
    # print(winkler_value)
    return winkler_value


def sim(s1, s2, p=0.6, l=0.1):
    if not isinstance(s1, (str, list, tuple)):
        s1 = str(s1)  # Convert to string if it's not
    if not isinstance(s2, (str, list, tuple)):
        s2 = str(s2)
    comm_value = commonality(s1, s2)
    diff_value = difference(s1, s2, p)
    winkler_value = winkler(s1, s2, comm_value, diff_value, l)
    similarity = comm_value - diff_value + winkler_value
    # print(similarity)
    return similarity


# Example usage
# s1 = "numberofpages"
# s2 = "noofpages"
# s1 = s1.lower()
# s2 = s2.lower()
# similarity = sim(s1, s2, p=0.6, l=0.1)
# print(f"Similarity between '{s1}' and '{s2}' is: {similarity}")
