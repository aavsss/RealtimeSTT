import React from 'react';
import { Flex, Typography } from 'antd';
import MainLayout from './MainLayout';
import TextAudioComponent from './TextAudioComponent';

const { Title, Paragraph } = Typography;

export default function Dashboard() {
  return (
    <MainLayout>
      <div className="App">
        <Flex vertical>
          <div style={{ margin: 'auto', textAlign: 'center' }}>
            <Title style={{ fontSize: '2.25rem' }}> Transcribe from Nepali to English</Title>
            <Paragraph
              style={{ margin: 'auto', fontSize: '20px', width: '50%' }}
            >
              Empower Your Voice, Transform Your Words: Lightning-Fast Speech to
              Text (STT) for a Seamless Communication
              Experience!
            </Paragraph>
          </div>
          <TextAudioComponent />
        </Flex>
      </div>
    </MainLayout>
  );
}
