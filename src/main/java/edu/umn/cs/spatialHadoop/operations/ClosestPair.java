/***********************************************************************
 * Copyright (c) 2015 by Regents of the University of Minnesota.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Apache License, Version 2.0 which
 * accompanies this distribution and is available at
 * http://www.opensource.org/licenses/apache2.0.php.
 *
 *************************************************************************/
package edu.umn.cs.spatialHadoop.operations;

import edu.umn.cs.spatialHadoop.OperationsParams;
import edu.umn.cs.spatialHadoop.core.Point;
import edu.umn.cs.spatialHadoop.core.Rectangle;
import edu.umn.cs.spatialHadoop.core.Shape;
import edu.umn.cs.spatialHadoop.core.SpatialSite;
import edu.umn.cs.spatialHadoop.mapred.TextOutputFormat3;
import edu.umn.cs.spatialHadoop.mapreduce.RTreeRecordReader3;
import edu.umn.cs.spatialHadoop.mapreduce.SpatialInputFormat3;
import edu.umn.cs.spatialHadoop.mapreduce.SpatialRecordReader3;
import edu.umn.cs.spatialHadoop.nasa.HDFRecordReader;
import edu.umn.cs.spatialHadoop.util.MemoryReporter;
import edu.umn.cs.spatialHadoop.util.Parallel;
import edu.umn.cs.spatialHadoop.util.Parallel.RunnableRange;
import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapred.Task;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.OutputCommitter;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.IndexedSortable;
import org.apache.hadoop.util.QuickSort;



import static jcuda.driver.JCudaDriver.*;

/**
 * Closest pair of points algorithm
 * @author Ahmed Eldawy
 *
 */
public class ClosestPair {

  /**Logger to write log messages for this class*/
  static final Log LOG = LogFactory.getLog(ClosestPair.class);

  public static final String BruteForceThreshold = "ClosestPair.BruteForceThreshold";


  /**
   * A pair of points.
   * @author Ahmed Eldawy
   *
   */
  public static class Pair {
    public Point p1, p2;

    public double getDistance() {
      return p1.distanceTo(p2);
    }

    @Override
    public String toString() {
      return String.format("Pair (%s, %s) - Distance(%f)", p1.toString(),
              p2.toString(), p1.distanceTo(p2));
    }
  }

  /**
   * Finds the closest pair using an in-memory divide and conquer algorithm.
   * @param points
   * @param threshold
   * @return
   */


  public static Pair closestPairInMemory(final Point[] points, int threshold) throws URISyntaxException {
    // Sort points by increasing x-axis

    Arrays.sort(points);

    float inp[] = new float[2*points.length];
    int ind = 0;
    // long t11 = System.currentTimeMillis();
    for(Point point : points){
      inp[ind] = (float)point.x;
      inp[ind+1] = (float)point.y;
      ind += 2;
    }
    
    // long t12 = System.currentTimeMillis();
    // System.out.println("Time taken to finish the for loop that initializes the input array to be sent to GPU - "+(t12-t11));

    class SubListComputation {
      int start, end;
      int p1, p2;
      double distance;
    }

    List<SubListComputation> sublists = new ArrayList<SubListComputation>();

    // Compute the closest pair for each sublist below the threshold
    int start = 0;

    int blocks = 0;
    // long t1 = System.currentTimeMillis();
    while(start < points.length){
      ++blocks;
      int end;
      if (start + (threshold * 3 / 2) > points.length)
        end = points.length;
      else
        end = start + threshold;
      start = end;
    }
    // long t2 = System.currentTimeMillis();
    //HERE I AM GETTING THE NUMBER OF WHILE LOOPS THAT WILL BE EXECUTED AND INITIALLY I THOUGHT I COULD USE AS MANY BLOCKS AND HENCE NAMED THE VARIABLE THAT KEEPS COUNT AS BLOCKS. BUT TURNS OUT THE NUMBER OF ITERATIONS WILL BE GREATER THAN THE NUM OF BLOCKS THAT WE CAN ALLOCATE TO A GPU.

    int startPoints[] = new int[blocks];
    int endPoints[] = new int[blocks];
    float distancePoints[] = new float[blocks];

    start = 0;
    ind = 0;
    // long t3 = System.currentTimeMillis();
    while(start < points.length){
      int end;
      if (start + (threshold * 3 / 2) > points.length)
        end = points.length;
      else
        end = start + threshold;
      startPoints[ind] = start;
      endPoints[ind] = end;
      distancePoints[ind] = (float)points[start].distanceTo(points[start+1]);
      start = end;
      ++ind;
    }
    // long t4 = System.currentTimeMillis();
    //HERE I AM GETTING THE STARTING VALUES FOR EACH WHILE LOOP (WHICH WILL BE EXECUTED PARALLELY ACCROSS BLOCKS) AND STORING THEM AND THEY WILL BE LATER PASSED TO THE GPU.
    // int blocksExecuted = 0;
    URL url = ClosestPair.class.getClassLoader().getResource("gpu_test.ptx");
    String ptxFileName = url.getPath();

    //DOING ALL THIS HERE BECAUSE IF DONE INSIDE THE FUNCTION THEN IT KEEPS INITIALISING INPUT OVER AND OVER. INSTEAD INITIALIZING IT ONCE HERE SO THAT WE CAN AVOID EXTRA FOR LOOP
    //TODO: VARY THE NUMBER OF BLOCKS AND FIND OPTIMUM. AND ALSO SEE WHAT ELSE CAN BE DONE TO CAUSE A SPEED UP BECAUSE CURRENTLY THE WORKING IS SLOW.
    cuInit(0);
    // System.out.println("Initialised  cuda device!");
    CUdevice device = new CUdevice();
    cuDeviceGet(device, 0);
    CUcontext cUcontext = new CUcontext();
    cuCtxCreate(cUcontext, 0, device);
    // System.out.println("Initialized context");
    CUmodule module = new CUmodule();
    cuModuleLoad(module, ptxFileName);
    CUfunction function = new CUfunction();

    cuModuleGetFunction(function, module, "gpu");
    CUdeviceptr deviceInput = new CUdeviceptr();
    // cuMemAlloc(deviceInput, points.length * Sizeof.POINTER);
    cuMemAlloc(deviceInput, inp.length*Sizeof.FLOAT);
    // long t5 = System.currentTimeMillis();
    cuMemcpyHtoD(deviceInput, Pointer.to(inp), inp.length*Sizeof.FLOAT);
    // long t6 = System.currentTimeMillis();
    // CUdeviceptr hostPointers[] = new CUdeviceptr[points.length];
    // System.out.println("calling kernel!");
    //HERE I AM EXECUTING 1000 BLOCKS AT A TIME. WE WILL HAVE TO VARY IT AND CHECK IF THERE IS ANY IMPROVEMENT. MAY ALSO HAVE TO ADD SOME CONDITIONS TO MAKE SURE THE CORRECT NUMBER OF BLOCKS ARE GETTING EXECUTED EVERYTIME TOO.
    float cudaOutput[] = ClosestPairMap.executeKernel(blocks, deviceInput, function, startPoints, endPoints, distancePoints);
    ind = 0;
    //HERE OUTPUT FROM THE FUNCTION THAT EXECUTES THE CUDA FUNCTION IS 2D BECAUSE IT CONTAINS THE OUTPUT FROM EACH BLOCK WHICH AGAIN WAS EXECUTING WHATEVER WAS THERE IN EACH WHILE LOOP
    // long t7 = System.currentTimeMillis();
    for(int i=0;i<blocks*3;i+=3) {
//          System.out.println("Adding output to the list");
      SubListComputation closestPair = new SubListComputation();
      closestPair.start = startPoints[ind];
      closestPair.end = endPoints[ind];
//        System.out.println(cudaOutput[i+0]+" "+cudaOutput[i+1] + " "+cudaOutput[i+2]);
      closestPair.p1 = (int) cudaOutput[i+0];
      closestPair.p2 = (int) cudaOutput[i+1];
      closestPair.distance = cudaOutput[i+2];
      sublists.add(closestPair);
      ++ind;
    }

    cuCtxDestroy(cUcontext);
    // long t8 = System.currentTimeMillis();

    // System.out.println("Size of sublists - "+sublists.size());

    // Merge each pair of adjacent sublists
    while (sublists.size() > 1) {
      List<SubListComputation> newSublists = new ArrayList<SubListComputation>();
      for (int ilist = 0; ilist < sublists.size() - 1; ilist += 2) {
        SubListComputation list1 = sublists.get(ilist);
        SubListComputation list2 = sublists.get(ilist+1);
        SubListComputation merged = new SubListComputation();
        merged.start = list1.start;
        merged.end = list2.end;
        // The closest pair of (list1 UNION list2) is either the closest pair
        // of list1, list2, or a new closest pair with one point in list1
        // and one point in list2
        double mindistance = Math.min(list1.distance, list2.distance);
        double xmin = points[list1.end - 1].x - mindistance;
        double xmax = points[list2.start].x + mindistance;
        int leftMargin = exponentialSearchLeft(points, list1.end, xmin);
        int rightMargin = exponentialSearchRight(points, list2.start, xmax);
        int minPointL = leftMargin, minPointR = list2.start;
        double minDistanceLR = points[minPointL].distanceTo(points[minPointR]);
        if (rightMargin - leftMargin < threshold) {
          // Use brute force technique
          for (int i1 = leftMargin; i1 < list1.end; i1++) {
            for (int i2 = list2.start; i2 < rightMargin; i2++) {
              double distance = points[i1].distanceTo(points[i2]);
              if (distance < mindistance) {
                minPointL = i1;
                minPointR = i2;
                minDistanceLR = distance;
              }
            }
          }
        } else {
          // Use a y-sort technique
          final int[] rPoints = new int[rightMargin - list2.start];
          for (int i = 0; i < rPoints.length; i++)
            rPoints[i] = i + list2.start;
          IndexedSortable ysort = new IndexedSortable() {
            @Override
            public void swap(int i, int j) {
              int temp = rPoints[i]; rPoints[i] = rPoints[j]; rPoints[j] = temp;
            }

            @Override
            public int compare(int i, int j) {
              double dy = points[rPoints[i]].y - points[rPoints[j]].y;
              if (dy < 0) return -1; if (dy > 0) return 1; return 0;
            }
          };
          new QuickSort().sort(ysort, 0, rPoints.length);
          int rpoint1 = 0, rpoint2 = 0;
          for (int ilPoint = leftMargin; ilPoint < list1.end; ilPoint++) {
            Point lPoint = points[ilPoint];
            while (rpoint1 < rPoints.length && lPoint.y - points[rPoints[rpoint1]].y > mindistance)
              rpoint1++;
            while (rpoint2 < rPoints.length && points[rPoints[rpoint2]].y - lPoint.y < mindistance)
              rpoint2++;
            for (int rpoint = rpoint1; rpoint < rpoint2; rpoint++) {
              double distance = lPoint.distanceTo(points[rPoints[rpoint]]);
              if (distance < minDistanceLR) {
                minPointL = ilPoint;
                minPointR = rPoints[rpoint];
                minDistanceLR = distance;
              }
            }
          }
        }

        if (minDistanceLR < mindistance) {
          // The closest pair is in the middle (between list1 and list2)
          merged.distance = minDistanceLR;
          merged.p1 = minPointL;
          merged.p2 = minPointR;
        } else if (list1.distance < list2.distance) {
          // The closest pair is in list1
          merged.distance = list1.distance;
          merged.p1 = list1.p1;
          merged.p2 = list1.p2;
        } else {
          // The closest pair is in list2
          merged.distance = list2.distance;
          merged.p1 = list2.p1;
          merged.p2 = list2.p2;
        }

        newSublists.add(merged);
      }
      sublists = newSublists;
    }
    // long t9 = System.currentTimeMillis();
    // System.out.println("Time taken to finish the first while loop that gets the number of blocks - "+(t2-t1));
    // System.out.println("Time taken to finish the second while loop loop that initializes startPoints and endPoints array - "+(t4-t3));
    // System.out.println("Time taken to transfer input from host to device - "+(t6-t5));
    // System.out.println("Time taken to finish the for loop that adds output values to the sublist - "+(t8-t7));
    // System.out.println("Time taken to finish the final while loop - "+(t9-t8));
    Pair closestPair = new Pair();
    closestPair.p1 = points[sublists.get(0).p1];
    closestPair.p2 = points[sublists.get(0).p2];
    return closestPair;
  }


  public static Pair closestPairInMemoryReducer(final Point[] points, int threshold) throws URISyntaxException {
    // Sort points by increasing x-axis

    Arrays.sort(points);

    class SubListComputation {
      int start, end;
      int p1, p2;
      double distance;
    }

    List<SubListComputation> sublists = new ArrayList<SubListComputation>();

    // Compute the closest pair for each sublist below the threshold
    int start = 0;

    // long t1 = System.currentTimeMillis();
    while (start < points.length) {
      ClosestPairMap.count++;
      int end;
      if (start + (threshold * 3 / 2) > points.length)
        end = points.length;
      else
        end = start + threshold;
      SubListComputation closestPair = new SubListComputation();
      closestPair.start = start;
      closestPair.end = end;
      closestPair.p1 = start;
      closestPair.p2 = start+1;
      closestPair.distance = points[start].distanceTo(points[start+1]);
      for (int i1 = start; i1 < end; i1++) {
        for (int i2 = i1 + 1; i2 < end; i2++) {
          double distance = points[i1].distanceTo(points[i2]);
          if (distance < closestPair.distance) {
            closestPair.p1 = i1;
            closestPair.p2 = i2;
            closestPair.distance = distance;
          }
        }
      }

      sublists.add(closestPair);
      start = end;
    }
    // long t2 = System.currentTimeMillis();
    // System.out.println("Size of sublists - "+sublists.size());
    // Merge each pair of adjacent sublists
    while (sublists.size() > 1) {
      List<SubListComputation> newSublists = new ArrayList<SubListComputation>();
      for (int ilist = 0; ilist < sublists.size() - 1; ilist += 2) {
        SubListComputation list1 = sublists.get(ilist);
        SubListComputation list2 = sublists.get(ilist+1);
        SubListComputation merged = new SubListComputation();
        merged.start = list1.start;
        merged.end = list2.end;
        // The closest pair of (list1 UNION list2) is either the closest pair
        // of list1, list2, or a new closest pair with one point in list1
        // and one point in list2
        double mindistance = Math.min(list1.distance, list2.distance);
        double xmin = points[list1.end - 1].x - mindistance;
        double xmax = points[list2.start].x + mindistance;
        int leftMargin = exponentialSearchLeft(points, list1.end, xmin);
        int rightMargin = exponentialSearchRight(points, list2.start, xmax);
        int minPointL = leftMargin, minPointR = list2.start;
        double minDistanceLR = points[minPointL].distanceTo(points[minPointR]);
        if (rightMargin - leftMargin < threshold) {
          // Use brute force technique
          for (int i1 = leftMargin; i1 < list1.end; i1++) {
            for (int i2 = list2.start; i2 < rightMargin; i2++) {
              double distance = points[i1].distanceTo(points[i2]);
              if (distance < mindistance) {
                minPointL = i1;
                minPointR = i2;
                minDistanceLR = distance;
              }
            }
          }
        } else {
          // Use a y-sort technique
          final int[] rPoints = new int[rightMargin - list2.start];
          for (int i = 0; i < rPoints.length; i++)
            rPoints[i] = i + list2.start;
          IndexedSortable ysort = new IndexedSortable() {
            @Override
            public void swap(int i, int j) {
              int temp = rPoints[i]; rPoints[i] = rPoints[j]; rPoints[j] = temp;
            }

            @Override
            public int compare(int i, int j) {
              double dy = points[rPoints[i]].y - points[rPoints[j]].y;
              if (dy < 0) return -1; if (dy > 0) return 1; return 0;
            }
          };
          new QuickSort().sort(ysort, 0, rPoints.length);
          int rpoint1 = 0, rpoint2 = 0;
          for (int ilPoint = leftMargin; ilPoint < list1.end; ilPoint++) {
            Point lPoint = points[ilPoint];
            while (rpoint1 < rPoints.length && lPoint.y - points[rPoints[rpoint1]].y > mindistance)
              rpoint1++;
            while (rpoint2 < rPoints.length && points[rPoints[rpoint2]].y - lPoint.y < mindistance)
              rpoint2++;
            for (int rpoint = rpoint1; rpoint < rpoint2; rpoint++) {
              double distance = lPoint.distanceTo(points[rPoints[rpoint]]);
              if (distance < minDistanceLR) {
                minPointL = ilPoint;
                minPointR = rPoints[rpoint];
                minDistanceLR = distance;
              }
            }
          }
        }

        if (minDistanceLR < mindistance) {
          // The closest pair is in the middle (between list1 and list2)
          merged.distance = minDistanceLR;
          merged.p1 = minPointL;
          merged.p2 = minPointR;
        } else if (list1.distance < list2.distance) {
          // The closest pair is in list1
          merged.distance = list1.distance;
          merged.p1 = list1.p1;
          merged.p2 = list1.p2;
        } else {
          // The closest pair is in list2
          merged.distance = list2.distance;
          merged.p1 = list2.p1;
          merged.p2 = list2.p2;
        }

        newSublists.add(merged);
      }
      sublists = newSublists;
    }
    // long t3 = System.currentTimeMillis();
    // System.out.println("Time taken for first while loop - " + (t2-t1));
    //System.out.println("Time taken for second while loop - " + (t3-t2));
    Pair closestPair = new Pair();
    closestPair.p1 = points[sublists.get(0).p1];
    closestPair.p2 = points[sublists.get(0).p2];
    return closestPair;
  }

  /**
   * Exponential search on the first point with x-coordinate larger than the
   * given xmin.
   * @param points
   * @param bound2
   * @param xmin
   * @return
   */
  static int exponentialSearchLeft(Point[] points, int bound2, double xmin) {
    int size = 1;
    while (bound2 - size > 0 && points[bound2 - size].x > xmin)
      size *= 2;
    int bound1 = Math.max(0, bound2 - size);
    // Binary search in the given boundary
    while (bound1 < bound2) {
      int m = (bound1 + bound2) / 2;
      if (points[m].x >= xmin)
        bound2 = m;
      else
        bound1 = m + 1;
    }
    return bound1;
  }

  /**
   * Exponential search on the first point with x-coordinate less than the
   * given xmax.
   * @param points Array of all points
   * @param bound1 The first item to start the search
   * @param xmax The value of x to searc for
   * @return
   */
  static int exponentialSearchRight(Point[] points, int bound1, double xmax) {
    int size = 1;
    while (bound1 + size <= points.length && points[bound1 + size - 1].x > xmax)
      size *= 2;
    int bound2 = Math.min(points.length, bound1 + size);
    // Binary search in the given boundary
    while (bound1 < bound2) {
      int m = (bound1 + bound2) / 2;
      if (points[m].x >= xmax)
        bound2 = m;
      else
        bound1 = m + 1;
    }
    return bound1;
  }

  /**
   * The map function computes the closest pair for a partition and returns all
   * points that can possibly contribute to the global closest pair. This
   * includes the closest pair found in this partition as well as all points
   * that are closer to the partition boudnary than the distance between the
   * closest pair.
   * @author Ahmed Eldawy
   *
   */
  public static class ClosestPairMap
          extends Mapper<Rectangle, Iterable<Point>, IntWritable, Point> {

    /**Boundaries of columns to split partitions*/
    private double[] columnBoundaries;

    public CUdeviceptr deviceInput;

    public CUfunction cUfunction;

    public CUcontext cUcontext;

    public static int count = 0;

    @Override
    protected void setup(Context context) throws IOException,
            InterruptedException {
      this.columnBoundaries = SpatialSite.getReduceSpace(context.getConfiguration());
    }

    @Override
    protected void map(Rectangle key, Iterable<Point> values, Context context)
            throws IOException, InterruptedException {
      IntWritable column = new IntWritable();
      List<Point> points = new ArrayList<Point>();

      ArrayList<CUdeviceptr> hostPointers = new ArrayList<>();


      //--------------

      for (Point point : values) {
        points.add(point.clone());
      }


      // System.out.println("Length of the points array : " + String.valueOf(points.size()));

//      LOG.info("\n\nCalling GPU Kernel Function!\n\n");
//      System.out.println("\n\nCalling GPU Kernel Function!\n\n");
//      try {
//          executeKernel(context, (int)Math.ceil(Math.sqrt((double)points.size())), points);
//        } catch (URISyntaxException e) {
//          // TODO Auto-generated catch block
//          e.printStackTrace();
//      }
      Pair pair = null;
      try {
        pair = closestPairInMemoryReducer(points.toArray(new Point[points.size()]),
                context.getConfiguration().getInt(BruteForceThreshold, 100));

      } catch (URISyntaxException e) {
        e.printStackTrace();
      }

      // Output the two closest points as well as all points within the minimum
      // distance of the partition boundary
      if (key.isValid()) {
        int col = Arrays.binarySearch(this.columnBoundaries, key.getCenterPoint().x);
        if (col < 0)
          col = -col - 1;
        column.set(col);

        double minDistance = pair.getDistance();
        Rectangle innerRectangle = key.buffer(-minDistance, -minDistance);
        for (Point p : points) {
          if (!innerRectangle.contains(p))
            context.write(column, p);
        }

        // Write p1 and p2 if they have not been written using the previous loop
        if (innerRectangle.contains(pair.p1))
          context.write(column, (Point) pair.p1);
        if (innerRectangle.contains(pair.p2))
          context.write(column, (Point) pair.p2);
      }
    }

    public static float[] executeKernel(int blocks, CUdeviceptr deviceInput, CUfunction function, int[] startPoints, int[] endPoints, float[] distancePoints) throws URISyntaxException{
//      Configuration conf = context.getConfiguration();
//      LOG.info("Using map context to set file system");
//      conf.set("fs.defaultFS", "hdfs://0.0.0.0:9000");

//      float[][] kernelOutput = new float[blocks][3];
      CUdeviceptr deviceOutput = new CUdeviceptr();
      cuMemAlloc(deviceOutput, blocks*3*Sizeof.FLOAT);

      float finalOutput[] = new float[blocks*3];
      cuMemcpyHtoD(deviceOutput, Pointer.to(finalOutput), blocks*3*Sizeof.FLOAT);
//      CUdeviceptr outputPointers[] = new CUdeviceptr[blocks];
//      System.out.println("Executing for loop with 'blocks' number of iterations");
//      long T1 = System.currentTimeMillis();
//      for(int i=0;i<blocks;++i){
//
//        kernelOutput[i][0] = (float)startPoints[i];
//        kernelOutput[i][1] = (float)(startPoints[i]+1);
//        kernelOutput[i][2] = distancePoints[i];
////          System.out.println("start points in execute kernel - "+ startPoints[i]);
//        outputPointers[i] = new CUdeviceptr();
//        cuMemAlloc(outputPointers[i], 3 * Sizeof.FLOAT);
//        cuMemcpyHtoD(outputPointers[i], Pointer.to(kernelOutput[i]),
//                3 * Sizeof.FLOAT);
//      }
//      long T2 = System.currentTimeMillis();
//      System.out.println("time taken for 'blocks' number of iterations - "+(T2-T1));
//      cuMemcpyHtoD(deviceOutput,Pointer.to(outputPointers), blocks * Sizeof.POINTER);
      CUdeviceptr deviceStartPoints = new CUdeviceptr();
      cuMemAlloc(deviceStartPoints, blocks*Sizeof.INT);

      CUdeviceptr deviceEndPoints = new CUdeviceptr();
      cuMemAlloc(deviceEndPoints, blocks*Sizeof.INT);

      CUdeviceptr deviceDistancePoints = new CUdeviceptr();
      cuMemAlloc(deviceDistancePoints, blocks*Sizeof.INT);

      cuMemcpyHtoD(deviceStartPoints, Pointer.to(startPoints), blocks*Sizeof.INT);
      cuMemcpyHtoD(deviceEndPoints, Pointer.to(endPoints), blocks*Sizeof.INT);
      cuMemcpyHtoD(deviceDistancePoints, Pointer.to(distancePoints), blocks*Sizeof.FLOAT);


      Pointer kernelParams = Pointer.to(
              Pointer.to(deviceInput),
              Pointer.to(deviceOutput),
              Pointer.to(deviceStartPoints),
              Pointer.to(deviceEndPoints),
              Pointer.to(deviceDistancePoints)
      );

      // System.out.println();
      // System.out.println();
      // long t1 = System.currentTimeMillis();
//        System.out.println("EXECUTING KERNEL FUNCTION...\n\n");
      cuLaunchKernel(function,
              blocks, 1, 1,      // Grid dimension
              1, 1, 1,      // Block dimension
              0, null,               // Shared memory size and stream
              kernelParams, null // Kernel- and extra parameters
      );
      cuMemcpyDtoH(Pointer.to(finalOutput), deviceOutput, blocks*3*Sizeof.FLOAT);
//            System.out.println("Output from the kernel : "+String.valueOf(kernelOutput[0][1]));
      cuCtxSynchronize();
      // long t2 = System.currentTimeMillis();

      // System.out.println("Done executing kernel!");
      // System.out.println("Time taken to execute kernel and transfer output from device to host- "+(t2-t1));
      return finalOutput;
    }

  }

  /**
   * The reduce
   * @author
   *
   */
  public static class ClosestPairReduce
          extends Reducer<IntWritable, Point, NullWritable, Point> {

    @Override
    protected void reduce(IntWritable dummyColumn, Iterable<Point> values,
                          Context context) throws IOException, InterruptedException {

      List<Point> points = new ArrayList<Point>();
      Rectangle mbr = new Rectangle(Double.MAX_VALUE, Double.MAX_VALUE,
              -Double.MAX_VALUE, -Double.MAX_VALUE);
      for (Point point : values) {
        points.add(point.clone());
        mbr.expand(point);
      }

      Pair pair = null;
      try {
        pair = closestPairInMemoryReducer(points.toArray(new Point[points.size()]),
                context.getConfiguration().getInt(BruteForceThreshold, 100));
      } catch (URISyntaxException e) {
        e.printStackTrace();
      }

      // Output the two closest points as well as all points within the minimum
      // distance of the partition boundary
      double minDistance = pair.getDistance();
      Rectangle innerRectangle = mbr.buffer(-minDistance, -minDistance);
      final NullWritable dummyNull = NullWritable.get();
      for (Point p : points) {
        if (!innerRectangle.contains(p))
          context.write(dummyNull, p);
      }

      // Write p1 and p2 if they have not been written using the previous loop
      if (innerRectangle.contains(pair.p1))
        context.write(dummyNull, (Point) pair.p1);
      if (innerRectangle.contains(pair.p2))
        context.write(dummyNull, (Point) pair.p2);
    }
  }

  public static class ClosestPairOutputCommitter extends FileOutputCommitter {

    private Path outPath;

    public ClosestPairOutputCommitter(Path outputPath, TaskAttemptContext task)
            throws IOException {
      super(outputPath, task);
      outPath = outputPath;
    }

    @Override
    public void commitJob(final JobContext context) throws IOException {
      super.commitJob(context);
      // Read all resulting files and combine them together
      final FileSystem fs = outPath.getFileSystem(context.getConfiguration());
      final FileStatus[] outFiles = fs.listStatus(outPath, SpatialSite.NonHiddenFileFilter);
      final Path[] inPaths = new Path[outFiles.length];
      for (int i = 0; i < outFiles.length; i++)
        inPaths[i] = outFiles[i].getPath();

      try {
        Pair closestPair =
                closestPairLocal(inPaths, new OperationsParams(context.getConfiguration()));
        final PrintStream ps = new PrintStream(fs.create(new Path(outPath, "finalResult")));
        System.out.println("\n\nPrinting final Output - ");
        ps.println(closestPair.p1+"\t"+closestPair.p2);
        ps.close();
      } catch (InterruptedException | URISyntaxException e) {
        e.printStackTrace();
      }

      // Delete all intermediate files
      for (FileStatus outFile : outFiles)
        fs.delete(outFile.getPath(), false);
    }
  }


  public static class ClosestPairOutputFormat extends TextOutputFormat3<NullWritable, Point> {
    @Override
    public synchronized OutputCommitter getOutputCommitter(
            TaskAttemptContext context) throws IOException {
      Path jobOutputPath = getOutputPath(context);
      return new ClosestPairOutputCommitter(jobOutputPath, context);
    }
  }

  public static Job closestPairMapReduce(Path[] inPaths, Path outPath,
                                         OperationsParams params)
          throws IOException, InterruptedException, ClassNotFoundException {
    Job job = new Job(params, "Closest Pair");
    job.setJarByClass(ClosestPair.class);
    Shape shape = params.getShape("shape");

    cuInit(0);
    // System.out.println("Initialised  cuda device!");
    CUdevice device = new CUdevice();
    cuDeviceGet(device, 0);
    CUcontext cUcontext = new CUcontext();
    // Set map and reduce
    job.setMapperClass(ClosestPairMap.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(shape.getClass());
    job.setReducerClass(ClosestPairReduce.class);

    // System.out.println("Adding classpath");
    job.addFileToClassPath(new Path("/user/user/gpu_test.ptx"));
    // System.out.println("Added gpu kernel in test folder to classpath");
    job.addFileToClassPath(new Path("/user/user/jcuda-10.1.0.jar"));
    job.addFileToClassPath(new Path("/user/user/jcuda-natives-10.1.0-linux-x86_64.jar"));
    // Set input and output
    job.setInputFormatClass(SpatialInputFormat3.class);
    SpatialInputFormat3.setInputPaths(job, inPaths);
    job.setOutputFormatClass(ClosestPairOutputFormat.class);
    TextOutputFormat.setOutputPath(job, outPath);

    // Set column boundaries to define the boundaries of each reducer
    SpatialSite.splitReduceSpace(job, inPaths, params);

    // Submit the job
    if (!params.getBoolean("background", false)) {
      job.waitForCompletion(params.getBoolean("verbose", false));
      if (!job.isSuccessful())
        throw new RuntimeException("Job failed!");
    } else {
      job.submit();
    }
    return job;
  }

  /**
   * Computes the closest pair using a local single-machine algorithm
   * (no MapReduce)
   * @param inPaths
   * @param params
   * @return
   * @throws IOException
   * @throws InterruptedException
   */
  public static Pair closestPairLocal(Path[] inPaths, final OperationsParams params)
          throws IOException, InterruptedException, URISyntaxException {
    if (params.getBoolean("mem", false))
      MemoryReporter.startReporting();
    // 1- Split the input path/file to get splits that can be processed
    // independently
    final SpatialInputFormat3<Rectangle, Point> inputFormat =
            new SpatialInputFormat3<Rectangle, Point>();
    Job job = Job.getInstance(params);
    SpatialInputFormat3.setInputPaths(job, inPaths);
    final List<InputSplit> splits = inputFormat.getSplits(job);
    final Point[][] allLists = new Point[splits.size()][];

    // 2- Read all input points in memory
    LOG.info("Reading points from "+splits.size()+" splits");
    List<Integer> numsPoints = Parallel.forEach(splits.size(), new RunnableRange<Integer>() {
      @Override
      public Integer run(int i1, int i2) {
        int numPoints = 0;
        for (int i = i1; i < i2; i++) {
          try {
            List<Point> points = new ArrayList<Point>();
            FileSplit fsplit = (FileSplit) splits.get(i);
            final RecordReader<Rectangle, Iterable<Point>> reader =
                    inputFormat.createRecordReader(fsplit, null);
            if (reader instanceof SpatialRecordReader3) {
              ((SpatialRecordReader3)reader).initialize(fsplit, params);
            } else if (reader instanceof RTreeRecordReader3) {
              ((RTreeRecordReader3)reader).initialize(fsplit, params);
            } else if (reader instanceof HDFRecordReader) {
              ((HDFRecordReader)reader).initialize(fsplit, params);
            } else {
              throw new RuntimeException("Unknown record reader");
            }
            while (reader.nextKeyValue()) {
              Iterable<Point> pts = reader.getCurrentValue();
              for (Point p : pts) {
                points.add(p.clone());
              }
            }
            reader.close();
            numPoints += points.size();
            allLists[i] = points.toArray(new Point[points.size()]);
          } catch (IOException e) {
            throw new RuntimeException("Error reading file", e);
          } catch (InterruptedException e) {
            throw new RuntimeException("Error reading file", e);
          }
        }
        return numPoints;
      }
    }, params.getInt("parallel", Runtime.getRuntime().availableProcessors()));

    int totalNumPoints = 0;
    for (int numPoints : numsPoints)
      totalNumPoints += numPoints;

    LOG.info("Read "+totalNumPoints+" points and merging into one list");
    Point[] allPoints = new Point[totalNumPoints];
    int pointer = 0;

    for (int iList = 0; iList < allLists.length; iList++) {
      System.arraycopy(allLists[iList], 0, allPoints, pointer, allLists[iList].length);
      pointer += allLists[iList].length;
      allLists[iList] = null; // To let the GC collect it
    }

    LOG.info("Computing closest-pair for "+allPoints.length+" points");
    Pair closestPair = closestPairInMemoryReducer(allPoints,
            params.getInt(BruteForceThreshold, 100));
    return closestPair;
  }

  public static Job closestPair(Path[] inFiles, Path outPath, OperationsParams params)
          throws IOException, InterruptedException, ClassNotFoundException, URISyntaxException {
      Pair pair = closestPairLocal(inFiles, params);
      System.out.println("Final answer - " + pair);
      return null;
  }

  private static void printUsage() {
    System.out.println("ClosestPair");
    System.out.println("Computes the closest pair of points in the input file");
    System.out.println("Parameters: (* marks required parameters)");
    System.out.println("<input file>: (*) Path to file that contains all shapes");
    System.out.println("shape:<s> - Type of shapes stored in the input file");
    System.out.println("-local - Implement a local machine algorithm (no MapReduce)");
  }

  /**
   * @param args
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static void main(String[] args)
          throws IOException, InterruptedException, ClassNotFoundException, URISyntaxException {
    GenericOptionsParser parser = new GenericOptionsParser(args);
    OperationsParams params = new OperationsParams(parser);

    if (!params.checkInputOutput()) {
      printUsage();
      System.exit(1);
    }

    Path[] inFiles = params.getInputPaths();
    Path outPath = params.getOutputPath();

    long t1 = System.currentTimeMillis();
    Job job = closestPair(inFiles, outPath, params);
    long t2 = System.currentTimeMillis();
    System.out.println("Total time: " + (t2 - t1) + " millis");
    if (job != null) {
      System.out.println("Input points: "+job.getCounters().findCounter(Task.Counter.MAP_INPUT_RECORDS).getValue());
      System.out.println("Map output points: "+job.getCounters().findCounter(Task.Counter.MAP_OUTPUT_RECORDS).getValue());
      System.out.println("Reduce output points: "+job.getCounters().findCounter(Task.Counter.REDUCE_OUTPUT_RECORDS).getValue());
    }
  }

}
