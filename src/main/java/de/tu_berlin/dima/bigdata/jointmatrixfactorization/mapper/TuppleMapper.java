package de.tu_berlin.dima.bigdata.jointmatrixfactorization.mapper;

import de.tu_berlin.dima.bigdata.jointmatrixfactorization.util.Util;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.MapStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactFloat;
import eu.stratosphere.pact.common.type.base.PactInteger;
import eu.stratosphere.pact.common.type.base.PactString;

/**
 * Input: rating file
 * Output: <userID, itemID, Rating>
 * 
 * @author titicaca
 *
 */
public class TuppleMapper extends MapStub{

	private final PactRecord outputRecord = new PactRecord();

	
	@Override
	public void map(PactRecord record, Collector<PactRecord> collector)
			throws Exception {
		
		String text = record.getField(0, PactString.class).toString();
		String[] tokens = Util.splitPrefTokens(text);
		int userID = Util.readID(tokens[Util.USER_ID_POS]);
		int itemID = Util.readID(tokens[Util.ITEM_ID_POS]);
		float rating = Util.readRate(tokens[Util.RATING_POS]);
		
		outputRecord.setField(0, new PactInteger(userID));
		outputRecord.setField(1, new PactInteger(itemID));
		outputRecord.setField(2, new PactFloat(rating));
		
		collector.collect(outputRecord);
		
	}
	
}