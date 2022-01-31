import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.StringUtils;

public class HashTagFiltering {
	private static final String dtPath = "/output/u_dt";
	private static final String htPath = "/output/u_ht";
	private static final String SPLITTER = "\t| ";
	private static final boolean matchCase = false;
	private static final double decayingFactor = (double) 1 / Math.pow(10, 4);

	public static class UniqueDateMapper extends Mapper<Object, Text, Text, IntWritable> {

		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String dtKey = value.toString().split(",")[0];
			StringTokenizer itr = new StringTokenizer(dtKey);

			SimpleDateFormat dtFmt = new SimpleDateFormat("MM/dd/yyyy");

			while (itr.hasMoreTokens()) {
				try {
					dtKey = dtFmt.format(dtFmt.parse(itr.nextToken()));
					word.set(dtKey);
				} catch (ParseException ex) {
					System.err.printf("Date Mapper: Invalid date format %s", dtKey);
					return;
				}
				context.write(word, one);
			}
		}
	}

	public static class UniqueDateReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
		private IntWritable result = new IntWritable();

		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}

	public static class UniqueHTMapper extends Mapper<Object, Text, Text, Text> {

		private Text hashTag = new Text();
		private Text dtValue = new Text();
		private String wordToMatch = "";
		private Configuration conf;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			conf = context.getConfiguration();
			wordToMatch = conf.get("hashtag.match", "");
		}
		
		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String dt_ht = (matchCase) ? value.toString() : value.toString().toLowerCase();
			if (wordToMatch.toString() != "" && dt_ht.indexOf(wordToMatch.toString()) == -1) {
				dt_ht = "";
			} else {
				int sep = value.find(",");
				if (sep < 0) {
					System.err.printf("HashTag Mapper: Not Enough Records for %s", dt_ht);
					return;
				}
				String dateValue = dt_ht.substring(0, sep).trim();
				String htKey = dt_ht.substring(1 + sep).trim();
				SimpleDateFormat dtFmt = new SimpleDateFormat("MM/dd/yyyy");

				try {
					dateValue = dtFmt.format(dtFmt.parse(dateValue));
					dtValue.set(dateValue);
				} catch (ParseException ex) {
					System.err.printf("HashTag Mapper: Invalid Date Format %s", dateValue);
					return;
				}

				hashTag.set(htKey);
				context.write(hashTag, dtValue);
			}
		}
	}

	public static class UniqueHTReducer extends Reducer<Text, Text, Text, Text> {
		private Text dateColl = new Text();

		@Override
		public void reduce(Text hashTagKey, Iterable<Text> dateValues, Context context)
				throws IOException, InterruptedException {
			String dates = "";
			for (Text dateValue : dateValues) {
				dates += ", " + dateValue.toString();
			}
			dateColl.set(dates.substring(1));
			context.write(hashTagKey, dateColl);
		}
	}

	public static class ScoreMapper extends Mapper<Object, Text, DoubleWritable, Text> {

		private DoubleWritable score = new DoubleWritable();
		private Text hashTag = new Text();

		private Set<String> uniqueDt = new HashSet<String>();

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			FetchDate();
		}

		private void FetchDate() {
			try {
				Path pt = new Path("hdfs://0.0.0.0:19000/" + dtPath + "/part-r-00000");// Location of file in HDFS
				FileSystem fs = FileSystem.get(new Configuration());
				BufferedReader readDate = new BufferedReader(new InputStreamReader(fs.open(pt)));

				String date = null;
				while ((date = readDate.readLine()) != null) {
					uniqueDt.add(date.split(SPLITTER)[0]);
				}
				readDate.close();
			} catch (IOException ioex) {
				System.err.println("Error while fetching date from the intermediate output '"
						+ StringUtils.stringifyException(ioex));
			}
		}

		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			double calcScore = 0;
			Iterator<String> uDates = uniqueDt.iterator();
			String[] dates = value.toString().split(",");
			while (uDates.hasNext()) {
				String currDate = uDates.next().trim();
				int freq = 0;
				for (String dateValue : dates) {
					if (dateValue.trim().equalsIgnoreCase(currDate)) {
						freq += 1;
					}
				}

				calcScore = (1 - decayingFactor) * calcScore + freq;
			}
			hashTag.set(key.toString());
			score.set(calcScore * (-1));
			context.write(score, hashTag);
		}
	}

	public static class ScoreReducer extends Reducer<DoubleWritable, Text, Text, DoubleWritable> {
		private int noOfHashTags;
		private Configuration conf;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			conf = context.getConfiguration();
			noOfHashTags = conf.getInt("hashtag.nht", -5);
		}

		@Override
		public void reduce(DoubleWritable score, Iterable<Text> hashTags, Context context)
				throws IOException, InterruptedException {
			DoubleWritable posScore = new DoubleWritable();
			for (Text hashTag : hashTags) {
				if (noOfHashTags == -5 || noOfHashTags > 0) {
					if (noOfHashTags > 0) --noOfHashTags;
					posScore.set(score.get() * (-1));
					context.write(hashTag, posScore);
				} else
					break;
			}
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration config = new Configuration();
		GenericOptionsParser genericOP = new GenericOptionsParser(args);
		String[] cla = genericOP.getRemainingArgs();
		boolean flag4Job1 = false;
		boolean flag4Job2 = false;
		int noOfHashTags = -5;
		String wordToMatch = "";

		try {
			for (int i = 2; i < cla.length; ++i) {
				if ("-nht".equals(cla[i])) {
					noOfHashTags = Integer.parseInt(cla[++i]);
				} else if ("-match".equals(cla[i])) {
					wordToMatch = cla[++i].toString();
					wordToMatch = matchCase ? wordToMatch : wordToMatch.toLowerCase();
				}
			}

			Job jobDate = Job.getInstance(config, "uniquedate");
			jobDate.setJarByClass(HashTagFiltering.class);
			jobDate.setMapperClass(UniqueDateMapper.class);
			jobDate.setCombinerClass(UniqueDateReducer.class);
			jobDate.setReducerClass(UniqueDateReducer.class);
			jobDate.setOutputKeyClass(Text.class);
			jobDate.setOutputValueClass(IntWritable.class);

			FileInputFormat.addInputPath(jobDate, new Path(cla[0]));
			FileOutputFormat.setOutputPath(jobDate, new Path(dtPath));

			flag4Job1 = jobDate.waitForCompletion(true);

			if (flag4Job1) {
				Job jobHT = Job.getInstance(config, "uniquehashtag");
				jobHT.setJarByClass(HashTagFiltering.class);
				jobHT.setMapperClass(UniqueHTMapper.class);
				jobHT.setReducerClass(UniqueHTReducer.class);
				jobHT.setOutputKeyClass(Text.class);
				jobHT.setOutputValueClass(Text.class);

				jobHT.getConfiguration().set("hashtag.match", wordToMatch);

				FileInputFormat.addInputPath(jobHT, new Path(cla[0]));
				FileOutputFormat.setOutputPath(jobHT, new Path(htPath));

				flag4Job2 = jobHT.waitForCompletion(true);
			} else
				System.exit(2);

			if (flag4Job2) {
				Job jobScore = Job.getInstance(config, "score");
				jobScore.setJarByClass(HashTagFiltering.class);
				jobScore.setMapperClass(ScoreMapper.class);
				jobScore.setReducerClass(ScoreReducer.class);
				jobScore.setInputFormatClass(KeyValueTextInputFormat.class);
				jobScore.setOutputKeyClass(DoubleWritable.class);
				jobScore.setOutputValueClass(Text.class);

				jobScore.getConfiguration().setInt("hashtag.nht", noOfHashTags);

				FileInputFormat.addInputPath(jobScore, new Path(htPath));
				FileOutputFormat.setOutputPath(jobScore, new Path(cla[1]));

				System.exit(jobScore.waitForCompletion(true) ? 0 : 1);
			} else
				System.exit(2);
		} catch (Exception ex) {
			System.err.println(
					"Usage: HashTagFiltering <input_path> <output_path> [-nht desired_number_of_hashtags] [-match desired_hashtag]");
			System.exit(2);
		}
	}
}