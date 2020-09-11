package com.wandb.java.sinfunc;

import com.wandb.client.WandbRun;
import org.json.JSONObject;

import java.io.IOException;

/**
 * Plots a sin function in wandb using the Java Client.
 */
public class Example {
    public static void main(String[] args) throws IOException, InterruptedException {
        System.out.println("Hello from wandb in java!");

        // Create custom config object for fun!
        JSONObject config = new JSONObject();
        config.put("configNumber", 100);
        config.put("configString", "Config string value!");

        // Using builder to create run object
        System.out.println("Creating wandb run object.");
        WandbRun run = new WandbRun.Builder()
                .withConfig(config)
                .build();

        // Print out url to monitor run on wandb
        run.printRunInfo();

        // Compute and log values for a sin function
        for(double i = 0.0; i < Math.PI; i += 0.1) {
            JSONObject log = new JSONObject();
            log.put("x1", i);
            log.put("x2", i*2);
            run.log(log);
        }

        // Finish the wandb run (this is required to be called at then end of your run.)
        run.finish();
    }
}
