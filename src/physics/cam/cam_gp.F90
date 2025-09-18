subroutine torch_inference(phys_state)
    ! CAM Types
    type(physics_state), intent(inout) :: phys_state(:)

    ! Torch Types
    type(torch_tensor) :: in_tensor, temp_tensor, q_tensor
    type(torch_tensor) :: temp_std_tensor, hum_std_tensor, rand_tensor
    type(torch_tensor) :: temp_pert_tensor, hum_pert_tensor

    ! Local arrays to hold data from phys_state
    real(8), allocatable :: phys_state_t_array(:,:,:)
    real(8), allocatable :: phys_state_q_array(:,:,:)

    integer :: tensor_layout_3d(3) = [3,2,1]
    integer :: stack_layout(4) = [4,3,2,1]

    ! Integers
    integer :: i, num_chunks, num_cols, num_levels

    ! --- Model Initialization (if needed) ---
    if (.not. temp_model_initialized) then
        call init_temp_model(temp_model)
        temp_model_initialized = .true.
    end if
    if (.not. hum_model_initialized) then
        call init_hum_model(hum_model)
        hum_model_initialized = .true.
    end if

    ! --- Get Dimensions & Prepare Host Arrays ---
    num_chunks = size(phys_state)
    num_cols = size(phys_state(1)%t, 1)
    num_levels = size(phys_state(1)%t, 2)
    allocate(phys_state_t_array(num_chunks, num_cols, num_levels))
    allocate(phys_state_q_array(num_chunks, num_cols, num_levels))
    do i = 1, num_chunks
        phys_state_t_array(i, :, :) = phys_state(i)%t
        phys_state_q_array(i, :, :) = phys_state(i)%q(:,:,1)
    end do

    ! --- Create Tensors on GPU ---
    call torch_tensor_from_array(temp_tensor, phys_state_t_array, tensor_layout_3d, torch_kCUDA)
    call torch_tensor_from_array(q_tensor, phys_state_q_array, tensor_layout_3d, torch_kCUDA)
    
    type(torch_tensor), dimension(2) :: stack_tensors
    stack_tensors(1) = temp_tensor
    stack_tensors(2) = q_tensor
    call torch_stack(in_tensor, stack_tensors, 3, stack_layout)

    call torch_randn_like(rand_tensor, temp_tensor)

    ! --- STEP 1: Perform ALL Model Inferences on Original Data ---
    ! Temperature model forward pass
    call torch_model_forward(temp_model, in_tensor, temp_std_tensor)

    ! Humidity model forward pass (using a slice of the original in_tensor)
    type(torch_tensor) :: in_tensor_hum_slice
    call torch_tensor_slice(in_tensor_hum_slice, in_tensor, 2, 0, 15)
    call torch_model_forward(hum_model, in_tensor_hum_slice, hum_std_tensor)

    ! --- STEP 2: Apply ALL Perturbations In-Place ---
    ! Temperature perturbation
    call torch_tensor_mul(temp_pert_tensor, rand_tensor, temp_std_tensor)
    call torch_tensor_add_(temp_tensor, temp_pert_tensor) ! In-place

    ! Humidity perturbation
    type(torch_tensor) :: q_slice, rand_slice
    call torch_tensor_slice(q_slice, q_tensor, 2, 0, 15)
    call torch_tensor_slice(rand_slice, rand_tensor, 2, 0, 15)
    call torch_tensor_mul(hum_pert_tensor, rand_slice, hum_std_tensor)
    call torch_tensor_add_(q_slice, hum_pert_tensor) ! In-place

    ! --- STEP 3: Copy Final Results Back to Host ---
    call torch_tensor_to_array(temp_tensor, phys_state_t_array)
    call torch_tensor_to_array(q_tensor, phys_state_q_array)
    
    do i = 1, num_chunks
        phys_state(i)%t = phys_state_t_array(i, :, :)
        phys_state(i)%q(:,:,1) = phys_state_q_array(i, :, :)
    end do

    ! --- STEP 4: Clean Up ALL Tensors ---
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
    
    ! --- Deallocate Host Arrays ---
    deallocate(phys_state_t_array)
    deallocate(phys_state_q_array)

end subroutine torch_inference
