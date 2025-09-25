program mpi_parrallel
  use mpi !调用mpi模块
  implicit none

  integer :: ierr, num_processors, my_processor_id

  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, num_processors, ierr)  ! 获取CPU数量
  call MPI_COMM_RANK(MPI_COMM_WORLD, my_processor_id, ierr)  ! 获取当前CPU的编号

  print *, 'Hello from processor ', my_processor_id, ' of ', num_processors

  call MPI_FINALIZE(ierr)

end program mpi_parrallel