program mpi_parallel_computation
  use mpi
  implicit none
  
  integer :: ierr, num_processors, my_processor_id
  integer :: i, start_index, end_index, chunk_size
  integer, parameter :: total_numbers = 1000000
  integer, dimension(total_numbers) :: numbers
  integer, dimension(:), allocatable :: local_numbers
  integer :: local_size
  real :: start_time, end_time, parallel_time
  integer :: status(MPI_STATUS_SIZE)  ! 正确的状态变量声明
  
  ! 初始化MPI
  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, num_processors, ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, my_processor_id, ierr)
  
  ! 同步所有进程，然后开始计时
  call MPI_BARRIER(MPI_COMM_WORLD, ierr)
  if (my_processor_id == 0) then
    call cpu_time(start_time)
  endif
  
  ! 只在主进程中打印一次hello
  if (my_processor_id == 0) then
    print *, 'hello'
  endif
  
  ! 计算每个进程应该处理的数据块大小
  chunk_size = total_numbers / num_processors
  start_index = my_processor_id * chunk_size + 1
  end_index = (my_processor_id + 1) * chunk_size
  
  ! 处理不能整除的情况
  if (my_processor_id == num_processors - 1) then
    end_index = total_numbers
  endif
  
  local_size = end_index - start_index + 1
  allocate(local_numbers(local_size))
  
  ! 每个进程并行计算自己的数据块
  do i = 1, local_size
    local_numbers(i) = (start_index + i - 1) + 10
  enddo
  
  ! 收集结果到主进程
  if (my_processor_id == 0) then
    numbers(1:chunk_size) = local_numbers
    do i = 1, num_processors - 1
      start_index = i * chunk_size + 1
      if (i == num_processors - 1) then
        end_index = total_numbers
        local_size = end_index - start_index + 1
      else
        end_index = (i + 1) * chunk_size
        local_size = chunk_size
      endif
      call MPI_RECV(numbers(start_index), local_size, MPI_INTEGER, &
                   i, 0, MPI_COMM_WORLD, status, ierr)
    enddo
  else
    call MPI_SEND(local_numbers, local_size, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, ierr)
  endif
  
  ! 计算并行时间
  call MPI_BARRIER(MPI_COMM_WORLD, ierr)
  if (my_processor_id == 0) then
    call cpu_time(end_time)
    parallel_time = end_time - start_time
    
    ! 输出结果和时间
    print *, '计算结果（前10个）:'
    do i = 1, min(10, total_numbers)
      write(*, '(I3, A, I3)') i, ' + 10 = ', numbers(i)
    enddo
    
    print *, '并行计算时间: ', parallel_time, '秒'
    print *, '使用的进程数: ', num_processors
    print *, '总计算量: ', total_numbers, '次运算'
  endif
  
  deallocate(local_numbers)
  call MPI_FINALIZE(ierr)

end program mpi_parallel_computation