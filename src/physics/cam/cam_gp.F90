module cam_gp
    use ftorch
    use physics_types,  only: physics_state
    implicit none

    public :: torch_inference

    ! Declare the torch models
    type(torch_model), save :: temp_model
    type(torch_model), save :: hum_model
    logical, save :: temp_model_initialized = .false.
    logical, save :: hum_model_initialized = .false.

contains

    subroutine init_temp_model(model)
        ! Initialize the temperature model
        type(torch_model), intent(inout) :: model
        call torch_model_load(model, "/weights/gp_temp.pt", torch_kCUDA)
    end subroutine init_temp_model

    subroutine init_hum_model(model)
        ! Initialize the humidity model
        type(torch_model), intent(inout) :: model
        call torch_model_load(model, "/weights/gp_hum.pt", torch_kCUDA)
    end subroutine init_hum_model

    subroutine torch_inference(phys_state)
        ! CAM Types
        type(physics_state), intent(inout) :: phys_state(:)

        ! Torch Types
        ! in_tensor: Stacked input tensor for the models (temp, q)
        ! temp_tensor, q_tensor: Tensors for temperature and specific humidity
        ! temp_std_tensor, hum_std_tensor: Output tensors holding standard deviations
        ! rand_tensor: Tensor of standard normal random numbers
        ! temp_pert_tensor, hum_pert_tensor: Tensors for the calculated perturbations
        type(torch_tensor) :: in_tensor, temp_tensor, q_tensor
        type(torch_tensor) :: temp_std_tensor, hum_std_tensor, rand_tensor
        type(torch_tensor) :: temp_pert_tensor, hum_pert_tensor

        ! Local arrays to hold data from phys_state
        real(8), allocatable :: phys_state_t_array(:,:,:)
        real(8), allocatable :: phys_state_q_array(:,:,:)
        real(8), allocatable :: new_phys_state_t_array(:,:,:)

        integer :: tensor_layout_3d(3) = [3,2,1]
        integer :: stack_layout(4) = [4,3,2,1] ! This is for the output of torch_stack, which will be 4D

        ! Integers
        integer :: i, num_chunks, num_cols, num_levels, num_species

        ! Initialize models if not already initialized
        if (.not. temp_model_initialized) then
            call init_temp_model(temp_model)
            temp_model_initialized = .true.
        end if
        if (.not. hum_model_initialized) then
            call init_hum_model(hum_model)
            hum_model_initialized = .true.
        end if

        ! Get dimensions from phys_state
        num_chunks = size(phys_state)
        num_cols = size(phys_state(1)%t, 1)
        num_levels = size(phys_state(1)%t, 2)
        num_species = size(phys_state(1)%q, 3)

        ! Allocate local arrays
        allocate(phys_state_t_array(num_chunks, num_cols, num_levels))
        ! We only need the specific humidity (first species)
        allocate(phys_state_q_array(num_chunks, num_cols, num_levels))
        allocate(new_phys_state_t_array(num_chunks, num_cols, num_levels))

        ! Fill local arrays from phys_state
        do i = 1, num_chunks
            phys_state_t_array(i, :, :) = phys_state(i)%t
            phys_state_q_array(i, :, :) = phys_state(i)%q(:,:,1) ! Specific humidity
        end do

        ! Create tensors from local arrays
        call torch_tensor_from_array(temp_tensor, phys_state_t_array, tensor_layout_3d, torch_kCUDA)
        call torch_tensor_from_array(q_tensor, phys_state_q_array, tensor_layout_3d, torch_kCUDA)

        ! Stack temperature and humidity tensors to create a single input tensor
        type(torch_tensor), dimension(2) :: stack_tensors
        stack_tensors(1) = temp_tensor
        stack_tensors(2) = q_tensor
        call torch_stack(in_tensor, stack_tensors, 3, stack_layout) ! Stack along a new last dimension

        ! Generate a tensor of standard normal random numbers
        call torch_randn_like(rand_tensor, temp_tensor)

        ! Perform inference and apply perturbations

        ! Temperature model
        call torch_model_forward(temp_model, in_tensor, temp_std_tensor)
        ! Scale the random numbers by the standard deviation to get the perturbation
        call torch_tensor_mul(temp_pert_tensor, rand_tensor, temp_std_tensor)
        ! Add the perturbation to the original temperature tensor
        call torch_tensor_add(temp_tensor, temp_tensor, temp_pert_tensor)
        call torch_tensor_to_array(temp_tensor, new_phys_state_t_array)

        ! Humidity model
        ! Create a slice of the input tensor for the humidity model (bottom 15 levels)
        type(torch_tensor) :: in_tensor_hum_slice
        call torch_tensor_slice(in_tensor_hum_slice, in_tensor, 2, 0, 15)
        call torch_model_forward(hum_model, in_tensor_hum_slice, hum_std_tensor)

        ! Get slices of the humidity and random tensors for the bottom 15 levels
        type(torch_tensor) :: q_slice, rand_slice
        call torch_tensor_slice(q_slice, q_tensor, 2, 0, 15) ! slice along the level dimension
        call torch_tensor_slice(rand_slice, rand_tensor, 2, 0, 15) ! slice along the level dimension

        ! Scale the random numbers by the standard deviation to get the perturbation
        call torch_tensor_mul(hum_pert_tensor, rand_slice, hum_std_tensor)
        ! Add the perturbation to the humidity slice in-place
        call torch_tensor_add_(q_slice, hum_pert_tensor) ! In-place add

        ! Copy the full q tensor (with the perturbed slice) back to the host
        call torch_tensor_to_array(q_tensor, phys_state_q_array)

        ! Copy the updated data back to phys_state
        do i = 1, num_chunks
            phys_state(i)%t = new_phys_state_t_array(i, :, :)
            ! Update only the specific humidity
            phys_state(i)%q(:,:,1) = phys_state_q_array(i, :, :)
        end do

        ! Free the torch tensors
        call torch_tensor_delete(in_tensor)
        call torch_tensor_delete(temp_tensor)
        call torch_tensor_delete(q_tensor)
        call torch_tensor_delete(temp_std_tensor)
        call torch_tensor_delete(hum_std_tensor)
        call torch_tensor_delete(rand_tensor)
        call torch_tensor_delete(temp_pert_tensor)
        call torch_tensor_delete(hum_pert_tensor)
        call torch_tensor_delete(q_slice)
        call torch_tensor_delete(rand_slice)
        call torch_tensor_delete(in_tensor_hum_slice)

        ! Deallocate local arrays
        deallocate(phys_state_t_array)
        deallocate(phys_state_q_array)
        deallocate(new_phys_state_t_array)

    end subroutine torch_inference

end module cam_gp
