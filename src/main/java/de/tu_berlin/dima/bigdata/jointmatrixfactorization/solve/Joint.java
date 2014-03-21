/*
 * Project: JointMatrixFactorization
 * @author Fangzhou Yang
 * @author Xugang Zhou
 * @version 1.0
 */

package de.tu_berlin.dima.bigdata.jointmatrixfactorization.solve;

import org.apache.mahout.math.Vector;

import de.tu_berlin.dima.bigdata.jointmatrixfactorization.type.PactVector;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.MatchStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactFloat;
import eu.stratosphere.pact.common.type.base.PactInteger;

/*
 * This Join class join each rating entry to a user- or item-feature-vector
 * In order to produce a entry contains the rating value and the feature vector 
 */
public class Joint extends MatchStub{

	private final PactRecord outputRecord = new PactRecord();
	private final PactVector vectorWritable = new PactVector();
	
	/*
	 * This override method defines how the join works
	 */
	@Override
	public void match(PactRecord ratingRecord, PactRecord featureRecord,
			Collector<PactRecord> collector) throws Exception {

		int userID = ratingRecord.getField(0, PactInteger.class).getValue();
		int itemID = ratingRecord.getField(1, PactInteger.class).getValue();
		float rating = ratingRecord.getField(2, PactFloat.class).getValue();
		Vector featureVector = featureRecord.getField(1, PactVector.class).get();
		vectorWritable.set(featureVector);
		
		outputRecord.setField(0, new PactInteger(userID));
		outputRecord.setField(1, new PactInteger(itemID));
		outputRecord.setField(2, new PactFloat(rating));
		outputRecord.setField(3, vectorWritable);
		
		collector.collect(outputRecord);
		
	}
	
}