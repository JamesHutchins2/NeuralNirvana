#include <stdlib.h>
#include <pthread.h>

typedef struct {
    int value;
    int from_sub_array;
    int next_index_in_sub_array;
} HeapNode;

typedef struct {
    int *array;
    int low;
    int high;
} ThreadData;

void quicksort(int *array, int low, int high) {
    if (low < high) {
        int pivot = partition(array, low, high); // Implement the partition logic
        quicksort(array, low, pivot - 1);
        quicksort(array, pivot + 1, high);
    }
}

int partition(int *array, int low, int high) {
    int pivot = array[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (array[j] < pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }
    swap(&array[i + 1], &array[high]);
    return (i + 1);
}

void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

void *thread_sort(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    quicksort(data->array, data->low, data->high);
    return NULL;
}

void parallel_sort(int *array, int size, int thread_count) {
    pthread_t threads[thread_count];
    ThreadData data[thread_count];

    // Calculate segment size and create threads
    int segment_size = size / thread_count;
    for (int i = 0; i < thread_count; ++i) {
        data[i].array = array;
        data[i].low = i * segment_size;
        data[i].high = (i == thread_count - 1) ? (size - 1) : (data[i].low + segment_size - 1);
        pthread_create(&threads[i], NULL, thread_sort, &data[i]);
    }

    // Wait for threads to finish
    for (int i = 0; i < thread_count; ++i) {
        pthread_join(threads[i], NULL);
    }

    // Merge the sorted segments
    // This step is complex and requires a custom implementation to merge multiple sorted arrays
    // One approach is to use a min-heap to efficiently merge sorted sub-arrays
    merge_segments(array, size, segment_size, thread_count);
}

void merge_segments(int *array, int size, int segment_size, int thread_count) {
    // Initialize the heap array and other variables
    HeapNode *heap = malloc(sizeof(HeapNode) * thread_count);
    int *merged_array = malloc(sizeof(int) * size);
    int i, j, heap_size = 0;

    // Function to build and maintain the min-heap will be needed here

    // Insert the first element of each sub-array into the heap
    for (i = 0; i < thread_count; ++i) {
        int index = i * segment_size;
        if (index < size) {
            heap[heap_size].value = array[index];
            heap[heap_size].from_sub_array = i;
            heap[heap_size].next_index_in_sub_array = index + 1;
            heap_size++;
            // Heapify the heap here
        }
    }

    // Merge the arrays
    for (j = 0; j < size; ++j) {
        // Extract the minimum element from the heap
        HeapNode min_node = heap[0];
        merged_array[j] = min_node.value;

        // Replace the minimum element with the next element of the same sub-array
        if (min_node.next_index_in_sub_array < (min_node.from_sub_array + 1) * segment_size && min_node.next_index_in_sub_array < size) {
            min_node.value = array[min_node.next_index_in_sub_array];
            min_node.next_index_in_sub_array++;
        } else {
            // If the sub-array is exhausted, replace it with the last element in the heap
            min_node = heap[heap_size - 1];
            heap_size--;
        }
        heap[0] = min_node;
        // Heapify the heap here
    }

    // Copy the merged array back to the original array
    for (i = 0; i < size; ++i) {
        array[i] = merged_array[i];
    }

    free(merged_array);
    free(heap);
}

void sort_array(int *array, int array_size, int thread_count) {
    // Ensure the number of threads does not exceed the array size
    if (thread_count > array_size) {
        thread_count = array_size;
    }

    // Call the parallel sort function
    parallel_sort(array, array_size, thread_count);
}