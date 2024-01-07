use super::execution_step::ExecutionStep;


pub struct Dot {
    elementwise_mul: ExecutionStep,
    block_sum_reduce: ExecutionStep,
    sum_reduce: ExecutionStep,
}