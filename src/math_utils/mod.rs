/// Returns ceil(log2(n))
pub fn log2_ceil(n: u32) -> u32 {
    let num_bits = std::mem::size_of::<u32>() as u32 * 8;
    let is_power_of_two = n.is_power_of_two() as u32;

    num_bits - n.leading_zeros() - is_power_of_two
}

/// Returns the next power of two greater than or equal to `n`.
pub fn next_closest_power_of_two(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }

    let mut result = 1;
    while result < n {
        result <<= 1;
    }

    result
}
