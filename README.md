Audio Classifier Using Statistical Learning
==================================================

Experimental project for voice command intention detection based on statistical learning
Java FFT and audio analysing took from [audio-analysis](https://github.com/Uriopass/audio-analysis)

## Testing

Execute the unit tests. More to come...

```kotlin
    val featureNames = mutableMapOf<String, Int>()
    for (i in 1..30) {
        featureNames["$i"] = i
    }
    
    // audio samples for training: 0.0 = YES 1.0 = NO
    val yes2 = audioToFs(0, "src/test/resources/yes2.wav", featureNames, 0.0)
    val yes3 = audioToFs(1, "src/test/resources/yes3.wav", featureNames, 0.0)
    val yes4 = audioToFs(2, "src/test/resources/yes4.wav", featureNames, 0.0)
    val no2 = audioToFs(3, "src/test/resources/no2.wav", featureNames, 1.0)
    val no3 = audioToFs(4, "src/test/resources/no3.wav", featureNames, 1.0)
    val no4 = audioToFs(5, "src/test/resources/no4.wav", featureNames, 1.0)
    
    
    val data = DataSet(arrayListOf(yes2, yes3, yes4, no2, no3, no4), featureNames.keys.toList())
    //println(data)
    
    val nb = NaiveBayes()
    
    // not known samples
    val yes1 = audioToFs(6, "src/test/resources/yes1.wav", featureNames, -1.0)
    val no1 = audioToFs(7, "src/test/resources/no1.wav", featureNames, -1.0)
    
    val stats = nb.train(data)
    
    //println(stats)
    
    val classification1 = nb.classify(yes1, stats)
    val classification2 = nb.classify(no1, stats)
    
    println(classification1.predictions)
    println(classification1.predict())
    
    println(classification2.predictions)
    println(classification2.predict())
    
    assertEquals(0.0, classification1.predict())
    assertEquals(1.0, classification2.predict())
```        
        