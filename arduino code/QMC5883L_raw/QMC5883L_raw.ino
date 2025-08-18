#include <Wire.h>
#include <QMC5883L.h>

QMC5883L mgntmtr;

int16_t mx, my, mz;
bool blinkState;
bool newData = true;
Vector raw;

void setup()
{
  /* Initialize UART communication */
  Serial.begin(38400); //Initializate Serial wo work well at 8MHz/16MHz

  /* Enable interrupts in digital pin 2 of Arduino */
  Serial.println("Enabling Arduino interruptions on digital pin 2...");
  attachInterrupt(digitalPinToInterrupt(2), sendDataFunction, RISING);

  /* Initialize device and check connection */ 
  Serial.println("Initialize QMC5883L...");
  if(!mgntmtr.begin())
  {
    Serial.println("QMC5883L connection failed");
    while(1);
  }

  Serial.println("QMC5883L connection succesful");
  
  /* Set measurement range */
  Serial.println("Setting measurement range...");
  // +/- 2.00 Ga: QMC5883L_RANGE_2GA
  // +/- 8.00 Ga: QMC5883L_RANGE_8GA 
  mgntmtr.setRange(QMC5883L_RANGE_2GA);

  /* Set measurement mode */
  Serial.println("Setting measurement mode...");
  // Single-Measurement:     QMC5883L_SINGLE
  // Continuous-Measurement: QMC5883L_CONTINOUS // (default)
  mgntmtr.setMeasurementMode(QMC5883L_CONTINOUS);

  /* Set number of samples averaged */
  Serial.println("Setting over sampling rate...");
  // 64 sample:  QMC5883L_OSR_64 (default)
  // 128 samples: QMC5883L_OSR_128
  // 256 samples: QMC5883L_OSR_256
  // 512 samples: QMC5883L_OSR_512
  mgntmtr.setOSR(QMC5883L_OSR_512);

  /* Set sample frequency */
  Serial.println("Setting sampling frequency...");
  // 10.00Hz: QMC5883L_DATARATE_10HZ (default)
  // 50.00Hz: QMC5883L_DATARATE_50HZ
  // 100.00Hz: QMC5883L_DATARATE_100HZ
  // 200.00Hz: QMC5883L_DATARATE_200HZ
  mgntmtr.setDataRate(QMC5883L_DATARATE_50HZ);

  /* Check settings */
  Serial.println("Checking settings...");
  checkSettings();

  /* Waiting for confirmation */
  uint8_t proceed = 0;
  while(!proceed){
    if (Serial.available() > 0) {
      proceed = 1;
    }
  }
  
  /* Procceding to get raw data */
  Serial.println("Getting raw data...");
  
  /*Configure board LED pin for output*/ 
  pinMode(LED_BUILTIN, OUTPUT);
}

void checkSettings()
{
  Serial.print("Selected range: ");
  
  switch (mgntmtr.getRange())
  {
    case QMC5883L_RANGE_2GA:  Serial.println("2.0 Ga"); break;
    case QMC5883L_RANGE_8GA:  Serial.println("8.0 Ga"); break;
    default: Serial.println("Bad range!");
  }
  
  Serial.print("Selected Measurement Mode: ");
  switch (mgntmtr.getMeasurementMode())
  {  
    case QMC5883L_SINGLE:  Serial.println("Single-Measurement"); break;
    case QMC5883L_CONTINOUS:  Serial.println("Continuous-Measurement"); break;
    default: Serial.println("Bad mode!");
  }
  
  Serial.print("Selected number of over sampling rate: ");
  switch (mgntmtr.getOSR())
  {  
    case QMC5883L_OSR_512 : Serial.println("512"); break;
    case QMC5883L_OSR_256 : Serial.println("256"); break;
    case QMC5883L_OSR_128 : Serial.println("128"); break;
    case QMC5883L_OSR_64 : Serial.println("64"); break;
    
    default: Serial.println("Bad number of over sampling rate!");
  }

  Serial.println("Selected sampling frequency: ");
  switch (mgntmtr.getDataRate())
  {  
    case QMC5883L_DATARATE_10HZ: Serial.println("Fs: 10.00 Hz"); break;
    case QMC5883L_DATARATE_50HZ:  Serial.println("Fs: 50.00 Hz"); break;
    case QMC5883L_DATARATE_100HZ:  Serial.println("Fs: 100.00 Hz"); break;
    case QMC5883L_DATARATE_200HZ: Serial.println("Fs: 200.00 Hz"); break;
    default: Serial.println("Bad data rate!");
  }
}

void loop()
{
  if (newData){
    raw = mgntmtr.readRaw();
    //Serial.print("mx:");
    Serial.print(raw.XAxis); 
    Serial.print(",");
    //Serial.print("my:");
    Serial.print(raw.YAxis); 
    Serial.print(",");
    //Serial.print("mz:");
    Serial.println(raw.ZAxis); 

    /* Clear the interrupt status */
    newData = false;
    /* Blink LED to indicate activity */
    blinkState = !blinkState;
    digitalWrite(LED_BUILTIN, blinkState);
  }
}

void sendDataFunction() {
  newData = true;
}
