program serial_computation
  implicit none
  
  integer :: i, result
  integer, parameter :: total_numbers = 1000000
  integer, dimension(total_numbers) :: numbers
  real :: start_time, end_time
  
  ! 开始计时
  call cpu_time(start_time)
  
  print *, 'hello'
  
  ! 串行计算
  do i = 1, total_numbers
    numbers(i) = i + 10
  enddo
  
  ! 结束计时
  call cpu_time(end_time)
  
  ! 输出结果（可选）
  print *, '计算结果（前10个）:'
  do i = 1, min(10, total_numbers)
    write(*, '(I3, A, I3)') i, ' + 10 = ', numbers(i)
  enddo
  
  print *, '串行计算时间: ', end_time - start_time, '秒'
  print *, '总计算量: ', total_numbers, '次运算'

end program serial_computation