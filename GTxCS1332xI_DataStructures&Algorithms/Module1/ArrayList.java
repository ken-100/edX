import java.util.NoSuchElementException;

/**
 * Your implementation of an ArrayList.
 */
public class ArrayList<T> {

    /*
     * The initial capacity of the ArrayList.
     *
     * DO NOT MODIFY THIS VARIABLE!
     */
    public static final int INITIAL_CAPACITY = 9;

    /*
     * Do not add new instance variables or modify existing ones.
     */
    private T[] backingArray;
    private int size;

    /**
     * This is the constructor that constructs a new ArrayList.
     *
     * Recall that Java does not allow for regular generic array creation,
     * so instead we cast an Object[] to a T[] to get the generic typing.
     */
    public ArrayList() {
        //DO NOT MODIFY THIS METHOD!
        backingArray = (T[]) new Object[INITIAL_CAPACITY];
        size = 0;
    }

    /**
     * Adds the data to the front of the list.
     *
     * This add may require elements to be shifted.
     *
     * Method should run in O(n) time.
     *
     * @param data the data to add to the front of the list
     * @throws java.lang.IllegalArgumentException if data is null
     */
    public void addToFront(T data) {
        // WRITE YOUR CODE HERE (DO NOT MODIFY METHOD HEADER)!
        if (data == null) {
            throw new IllegalArgumentException("Data is null");
        }
        if (size < backingArray.length) {
            for (int i=size-1; i >= 0; i--) {
                backingArray[i+1] = backingArray[i];
            }
            backingArray[0] = data;
            size += 1;
        } else if (size == backingArray.length) {
            T[] newArray = (T[]) new Object[backingArray.length * 2];
            newArray[0] = data;
            for (int i=0; i < backingArray.length; i++) {
                newArray[i+1] = backingArray[i];
            }
            newArray[0] = data;
            backingArray = newArray;
            size += 1;
        }
    }

    /**
     * Adds the data to the back of the list.
     *
     * Method should run in amortized O(1) time.
     *
     * @param data the data to add to the back of the list
     * @throws java.lang.IllegalArgumentException if data is null
     */
    public void addToBack(T data) {
        // WRITE YOUR CODE HERE (DO NOT MODIFY METHOD HEADER)!
        if (data == null) {
            throw new IllegalArgumentException("Data is null");
        }
        if (size < backingArray.length) {
            backingArray[size] = data;
            size++;
        } else if (size == backingArray.length) {
            T[] newArray = (T[]) new Object[backingArray.length * 2];
            for (int i=0; i < backingArray.length; i++) {
                newArray[i] = backingArray[i];
            }
            newArray[size] = data;
            size += 1;
            backingArray = newArray;
        }
    }

    /**
     * Removes and returns the first data of the list.
     *
     * Do not shrink the backing array.
     *
     * This remove may require elements to be shifted.
     *
     * Method should run in O(n) time.
     *
     * @return the data formerly located at the front of the list
     * @throws java.util.NoSuchElementException if the list is empty
     */
    public T removeFromFront() {
        // WRITE YOUR CODE HERE (DO NOT MODIFY METHOD HEADER)!
        if (size == 0) {
            throw new NoSuchElementException("No element to remove");
        }
        T frontData = backingArray[0];
        for (int i=0; i < size-1; i++) {
            backingArray[i] = backingArray[i+1];
        }
        backingArray[size-1] = null;
        size -= 1;
        return frontData;
    }

    /**
     */
    public T removeFromBack() {
        // WRITE YOUR CODE HERE (DO NOT MODIFY METHOD HEADER)!
        if (size == 0) {
            throw new NoSuchElementException("No element to remove");
        }
        T backData = backingArray[size-1];
        backingArray[size-1] = null;
        size -= 1;
        return backData;
    }

    /**
     */
    public T[] getBackingArray() {
        // DO NOT MODIFY THIS METHOD!
        return backingArray;
    }

    /**
     */
    public int size() {
        // DO NOT MODIFY THIS METHOD!
        return size;
    }
}
