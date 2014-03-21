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
 * This Reduce class reduce all entries with same userID to its user-feature-vector
 */
public class UserFeatureVectorUpdateReducer extends ReduceStub{
	
	PactRecord outputRecord = new PactRecord();
    /*
     * Get common variables
     */
	private final double lambda = Util.lambda;
	private final int numFeatures = Util.numFeatures;
	private final int numItems = Util.numItems;
	private final PactVector userFeatureVectorWritable = new PactVector();
	
	private static final Logger LOGGER = Logger.getLogger(UserFeatureVectorUpdateReducer.class.getName()); 

	
	/*
	 * This override method defines how user-feature-vector is calculated from all items' rating and their feature-vector
	 * @param in:Iterator[(userID, itemID, rating, user-feature-vector)] List with same userID
	 * @return (userID, user-feature-vector)
	 */
	@Override
	public void reduce(Iterator<PactRecord> records, Collector<PactRecord> collector)
			throws Exception {
		
		PropertyConfigurator.configure("log4j.properties");

		PactRecord currentRecord = null;
	    /*
	     * Set vector for put in the item rating from the user
	     * The itemID starts from 1
	     * So the initialized cardinality would be set to numItems + 1
	     */
		Vector vector = new RandomAccessSparseVector(numItems + 1, numItems + 1);
		int userID = -1;
		
	    /*
	     * Set a Map for all items' feature vectors
	     */
		OpenIntObjectHashMap<Vector> itemFeatureMatrix = new OpenIntObjectHashMap<Vector>(numItems);

	    /*
	     * Put all items' rating in a vector
	     * HashMap all the items' feature-vectors
	     */
		while (records.hasNext()) {
			currentRecord = records.next();
			
			userID = currentRecord.getField(0, PactInteger.class).getValue();
			int itemID = currentRecord.getField(1, PactInteger.class).getValue();
			float rating = currentRecord.getField(2, PactFloat.class).getValue();
			
			vector.setQuick(itemID, rating);
			
			Vector itemFeatureVector = currentRecord.getField(3, PactVector.class).get();
			
			itemFeatureMatrix.put(itemID, itemFeatureVector);
			
		}
		
	    /*
	     * Extract all item-feature-vectors whose itemID rated by the user
	     */    
		Vector userRatingVector = new SequentialAccessSparseVector(vector);
				
		List<Vector> featureVectors = Lists.newArrayListWithCapacity(userRatingVector.getNumNondefaultElements());
	    for (Vector.Element e : userRatingVector.nonZeroes()) {
	      int index = e.index();
	      if(itemFeatureMatrix.containsKey(index)){
	    	  featureVectors.add(itemFeatureMatrix.get(index));	  
	      }else{
	    	  System.out.println("Error! no such item:" + index +" in itemFeatureMatrix");
	    	  LOGGER.debug("Error! no such item:" + index +" in itemFeatureMatrix");
	      }
	    }
	    
	    if(userID > 0 ){
	        /*
	         * Calculate the user-feature-vector using Alternative Least Square (ALS) method
	         */
		    Vector userFeatureVector = AlternatingLeastSquaresSolver.solve(featureVectors, userRatingVector, lambda, numFeatures);
			userFeatureVectorWritable.set(userFeatureVector);
			outputRecord.setField(0, new PactInteger(userID));
			outputRecord.setField(1, userFeatureVectorWritable);
			collector.collect(outputRecord);
	    }
	    else{
	    	LOGGER.debug("Error! userID:" + userID + "muss be greater than zero!");
	    }

		
	}
	
}