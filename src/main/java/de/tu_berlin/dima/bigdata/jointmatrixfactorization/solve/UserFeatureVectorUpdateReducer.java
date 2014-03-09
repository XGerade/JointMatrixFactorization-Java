package de.tu_berlin.dima.bigdata.jointmatrixfactorization.solve;

import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import com.google.common.collect.Lists;

import de.tu_berlin.dima.bigdata.jointmatrixfactorization.type.PactVector;
import de.tu_berlin.dima.bigdata.jointmatrixfactorization.util.Util;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.ReduceStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactFloat;
import eu.stratosphere.pact.common.type.base.PactInteger;

/**
 * Input: <userID, itemID, Rating, featureVector>
 * Reduce Key: userID
 * Output: <userID, featureVector>
 * 
 * @author titicaca
 *
 */
public class UserFeatureVectorUpdateReducer extends ReduceStub{
	
	PactRecord outputRecord = new PactRecord();
	private final double lambda = Util.lambda;
	private final int numFeatures = Util.numFeatures;
	private final int numItems = Util.numItems;
	private final PactVector userFeatureVectorWritable = new PactVector();
	
	private static final Logger LOGGER = Logger.getLogger(UserFeatureVectorUpdateReducer.class.getName()); 

	@Override
	public void reduce(Iterator<PactRecord> records, Collector<PactRecord> collector)
			throws Exception {
		
		PropertyConfigurator.configure("log4j.properties");

		PactRecord currentRecord = null;
//		Vector userRatingVector = new SequentialAccessSparseVector();
		Vector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, numItems);
		int userID = -1;
		
		OpenIntObjectHashMap<Vector> itemFeatureMatrix = numItems > 0
		        ? new OpenIntObjectHashMap<Vector>(numItems) : new OpenIntObjectHashMap<Vector>();

		while (records.hasNext()) {
			currentRecord = records.next();
			
			userID = currentRecord.getField(0, PactInteger.class).getValue();
			int itemID = currentRecord.getField(1, PactInteger.class).getValue();
			float rating = currentRecord.getField(2, PactFloat.class).getValue();
			
			vector.setQuick(itemID, rating);
			
			Vector itemFeatureVector = currentRecord.getField(3, PactVector.class).get();
			
			itemFeatureMatrix.put(itemID, itemFeatureVector);
			
		}
		
		Vector userRatingVector = new SequentialAccessSparseVector(vector);
				
		List<Vector> featureVectors = Lists.newArrayListWithCapacity(userRatingVector.getNumNondefaultElements());
	    for (Vector.Element e : userRatingVector.nonZeroes()) {
	      int index = e.index();
	      if(itemFeatureMatrix.containsKey(index)){
	    	  featureVectors.add(itemFeatureMatrix.get(index));	  
//	    	  LOGGER.info("get itemID: " + index +" in userRatingVector of User " + userID);
	      }else{
	    	  System.out.println("Error! no such item:" + index +" in itemFeatureMatrix");
	    	  LOGGER.debug("Error! no such item:" + index +" in itemFeatureMatrix");
	      }
	    }
	    
	    if(userID > 0 ){
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