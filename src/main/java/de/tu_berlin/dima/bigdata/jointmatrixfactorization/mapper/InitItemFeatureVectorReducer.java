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

/**
 * Input: <userID, itemID, Rating> from TuppleMapper
 * Reduce Key: itemID
 * Output: <itemID, initFeatureVector>
 * 
 * @author titicaca
 *
 */

public class InitItemFeatureVectorReducer extends ReduceStub{

	private final Vector features = new SequentialAccessSparseVector(Integer.MAX_VALUE, Util.numFeatures);
	
	private final PactRecord outputRecord = new PactRecord();
	private final int numFeatures = Util.numFeatures;
	private final Random random = new Random();
	private final PactVector featureVector = new PactVector();

	
	@Override
	public void reduce(Iterator<PactRecord> records, Collector<PactRecord> collector)
			throws Exception {

		int itemID = -1;
		if(records.hasNext()){
			itemID = records.next().getField(1, PactInteger.class).getValue();	
		}
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