/*
 * Project: JointMatrixFactorization
 * @author Fangzhou Yang
 * @author Xugang Zhou
 * @version 1.0
 */

package de.tu_berlin.dima.bigdata.jointmatrixfactorization.mapper;

import java.util.Iterator;
import java.util.Random;

import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import de.tu_berlin.dima.bigdata.jointmatrixfactorization.type.PactVector;
import de.tu_berlin.dima.bigdata.jointmatrixfactorization.util.Util;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.ReduceStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactInteger;

/*
 * This Reduce class reduce all the input entries to each itemID and a random feature-vector 
 */
public class InitItemFeatureVectorReducer extends ReduceStub{

	private final Vector features = new SequentialAccessSparseVector(Integer.MAX_VALUE, Util.numFeatures);
	
	private final PactRecord outputRecord = new PactRecord();
	private final int numFeatures = Util.numFeatures;
	private final Random random = new Random();
	private final PactVector featureVector = new PactVector();

	/*
	 * This override method defines how the entries with same itemID reduce to a random feature-vector
	 * @param in:Iterator[(userID, itemID, rating)] List of entries with same itemID
	 * @return (itemID, item-feature-vector)
	 */
	@Override
	public void reduce(Iterator<PactRecord> records, Collector<PactRecord> collector)
			throws Exception {

		int itemID = -1;
		if(records.hasNext()){
			itemID = records.next().getField(1, PactInteger.class).getValue();	
		}
	    /*
	     * Give each feature a random value
	     */
		if(itemID > 0){
			for(int i = 0; i < numFeatures; i ++){
				features.set(i, random.nextFloat());
			}
			featureVector.set(features);
			
			outputRecord.setField(0, new PactInteger(itemID));
			outputRecord.setField(1, featureVector);
			
			collector.collect(outputRecord);
		}
		
	}
	
}