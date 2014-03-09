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
 * Reduce Key: itemID
 * Output: <userID, featureVector>
 * 
 * @author titicaca
 *
 */
public class ItemFeatureVectorUpdateReducer extends ReduceStub{
	
	PactRecord outputRecord = new PactRecord();
	private final double lambda = Util.lambda;
	private final int numFeatures = Util.numFeatures;
	private final int numUsers = Util.numUsers;
	private final PactVector itemFeatureVectorWritable = new PactVector();
	
	private static final Logger LOGGER = Logger.getLogger(ItemFeatureVectorUpdateReducer.class.getName()); 

	@Override
	public void reduce(Iterator<PactRecord> records, Collector<PactRecord> collector)
			throws Exception {
		
		PropertyConfigurator.configure("log4j.properties");

		PactRecord currentRecord = null;
//		Vector itemRatingVector = new SequentialAccessSparseVector();
		Vector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, numUsers);
		int itemID = -1;
		
		OpenIntObjectHashMap<Vector> userFeatureMatrix = numUsers > 0
		        ? new OpenIntObjectHashMap<Vector>(numUsers) : new OpenIntObjectHashMap<Vector>();

		while (records.hasNext()) {
			currentRecord = records.next();
			
			itemID = currentRecord.getField(1, PactInteger.class).getValue();
			int userID = currentRecord.getField(0, PactInteger.class).getValue();
			float rating = currentRecord.getField(2, PactFloat.class).getValue();
			
			vector.setQuick(userID, rating);
			
			Vector userFeatureVector = currentRecord.getField(3, PactVector.class).get();
			
			userFeatureMatrix.put(userID, userFeatureVector);
			
		}
				
		Vector itemRatingVector = new SequentialAccessSparseVector(vector);
		
		List<Vector> featureVectors = Lists.newArrayListWithCapacity(itemRatingVector.getNumNondefaultElements());
	    for (Vector.Element e : itemRatingVector.nonZeroes()) {
	      int index = e.index();
	      if(userFeatureMatrix.containsKey(index)){
	    	  featureVectors.add(userFeatureMatrix.get(index));	  
//	    	  LOGGER.info("get itemID: " + index +" in userRatingVector of User " + userID);
	      }else{
	    	  System.out.println("Error! no such item:" + index +" in itemFeatureMatrix");
	    	  LOGGER.debug("Error! no such item:" + index +" in itemFeatureMatrix");
	      }
	    }
	    
	    if(itemID > 0 ){
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