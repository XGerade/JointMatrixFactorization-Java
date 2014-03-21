/*
 * Project: JointMatrixFactorization
 * @author Fangzhou Yang
 * @author Xugang Zhou
 * @version 1.0
 */

package de.tu_berlin.dima.bigdata.jointmatrixfactorization.solve;

import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import com.google.common.collect.Lists;

import de.tu_berlin.dima.bigdata.jointmatrixfactorization.type.PactVector;
import de.tu_berlin.dima.bigdata.jointmatrixfactorization.util.Util;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.ReduceStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactFloat;
import eu.stratosphere.pact.common.type.base.PactInteger;

/*
 * This Reduce class reduce all entries with same itemID to its item-feature-vector
 */
public class ItemFeatureVectorUpdateReducer extends ReduceStub{
	
	PactRecord outputRecord = new PactRecord();
    /*
     * Get common variables
     */
	private final double lambda = Util.lambda;
	private final int numFeatures = Util.numFeatures;
	private final int numUsers = Util.numUsers;
	private final PactVector itemFeatureVectorWritable = new PactVector();
	
	private static final Logger LOGGER = Logger.getLogger(ItemFeatureVectorUpdateReducer.class.getName()); 

	/*
	 * This override method defines how item-feature-vector is calculated from all users' rating and their feature-vector
	 * @param in:Iterator[(userID, itemID, rating, user-feature-vector)] List with same itemID
	 * @return (itemID, item-feature-vector)
	 */
	@Override
	public void reduce(Iterator<PactRecord> records, Collector<PactRecord> collector)
			throws Exception {
		
		PropertyConfigurator.configure("log4j.properties");

	    /*
	     * Set vector for put in the user rating of the item
	     * The userID starts from 1
	     * So the initialized cardinality would be set to numUsers + 1
	     */
		PactRecord currentRecord = null;
		Vector vector = new RandomAccessSparseVector(numUsers + 1, numUsers + 1);
		int itemID = -1;
		
	    /*
	     * Set a Map for all users' feature vectors
	     */
		OpenIntObjectHashMap<Vector> userFeatureMatrix = new OpenIntObjectHashMap<Vector>(numUsers);

	    /*
	     * Put all users' rating in a vector
	     * HashMap all the users' feature-vectors
	     */
		while (records.hasNext()) {
			currentRecord = records.next();
			
			itemID = currentRecord.getField(1, PactInteger.class).getValue();
			int userID = currentRecord.getField(0, PactInteger.class).getValue();
			float rating = currentRecord.getField(2, PactFloat.class).getValue();
			
			vector.setQuick(userID, rating);
			
			Vector userFeatureVector = currentRecord.getField(3, PactVector.class).get();
			
			userFeatureMatrix.put(userID, userFeatureVector);
			
		}
				
	    /*
	     * Extract all user-feature-vectors whose userID rated for the item
	     */
		Vector itemRatingVector = new SequentialAccessSparseVector(vector);
		List<Vector> featureVectors = Lists.newArrayListWithCapacity(itemRatingVector.getNumNondefaultElements());
	    for (Vector.Element e : itemRatingVector.nonZeroes()) {
	      int index = e.index();
	      if(userFeatureMatrix.containsKey(index)){
	    	  featureVectors.add(userFeatureMatrix.get(index));	  
	      }else{
	    	  System.out.println("Error! no such item:" + index +" in itemFeatureMatrix");
	    	  LOGGER.debug("Error! no such item:" + index +" in itemFeatureMatrix");
	      }
	    }
	    
	    if(itemID > 0 ){
	        /*
	         * Calculate the item-feature-vector using Alternative Least Square (ALS) method
	         */
		    Vector itemFeatureVector = AlternatingLeastSquaresSolver.solve(featureVectors, itemRatingVector, lambda, numFeatures);
			itemFeatureVectorWritable.set(itemFeatureVector);
			outputRecord.setField(0, new PactInteger(itemID));
			outputRecord.setField(1, itemFeatureVectorWritable);
			collector.collect(outputRecord);
	    }
	    else{
	    	LOGGER.debug("Error! itemID:" + itemID + "muss be greater than zero!");
	    }

		
	}
	
}