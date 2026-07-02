import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Button,
  Collapse,
  DatePicker,
  Descriptions,
  Empty,
  Form,
  Grid,
  Input,
  Layout,
  Menu,
  Message,
  Modal,
  Popconfirm,
  Select,
  Space,
  Spin,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload,
} from "@arco-design/web-react";
import {
  IconDelete,
  IconEdit,
  IconFileAudio,
  IconHome,
  IconRefresh,
  IconSettings,
  IconUpload,
} from "@arco-design/web-react/icon";

const { Sider, Content } = Layout;
const { Title, Text } = Typography;
const { Row, Col } = Grid;
const FormItem = Form.Item;
const TextArea = Input.TextArea;

const PROCESSING_STATUSES = new Set([
  "uploaded",
  "queued_transcription",
  "transcribing",
  "diarizing",
  "queued_note_generation",
  "generating_note",
]);

const DEFAULT_SETTINGS: Settings = {
  data_root: "data",
  whisper_command:
    ".venv/bin/python scripts/transcribe_faster_whisper.py --audio {audio} --output {transcript_json} --model medium",
  diarization_command:
    ".venv/bin/python scripts/diarize_voice_features.py --audio {audio} --transcript-json {transcript_json} --output {diarization_json}",
  llm_command: ".venv/bin/python scripts/generate_note_ollama.py --prompt-file {prompt_file} --model qwen2.5:7b-instruct",
  clinical_prompt_path: "prompts/clinical_note_prompt.md",
};

type ViewName = "records" | "session" | "settings";
type Speaker = "Therapist" | "Client" | "Unknown";

type CaseItem = {
  case_id: string;
  note?: string;
};

type SessionSummary = {
  case_id: string;
  session_id: string;
  session_date?: string;
  file_name?: string;
  status?: string;
  progress_label?: string;
  error_message?: string;
  is_processing?: boolean;
  next_action?: string;
};

type TranscriptSegment = {
  start: number;
  end: number;
  speaker?: Speaker | string;
  text?: string;
  [key: string]: unknown;
};

type ClinicalNote = {
  session_summary?: Record<string, string[]>;
  soap?: Record<string, string[]>;
};

type SessionDetail = SessionSummary & {
  diarization_configured?: boolean;
  transcript?: TranscriptSegment[];
  clinical_note?: ClinicalNote;
  audio_path?: string;
  transcript_path?: string;
  note_json_path?: string;
  markdown_path?: string;
};

type Settings = {
  data_root: string;
  whisper_command: string;
  diarization_command: string;
  llm_command: string;
  clinical_prompt_path: string;
};

type UploadValues = {
  case_id: string;
  session_date: string;
  note?: string;
  audio?: Array<{ originFile?: File }>;
};

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || response.statusText);
  }
  return response.json();
}

function today() {
  return new Date().toISOString().slice(0, 10);
}

function statusColor(status?: string) {
  if (status === "complete") return "green";
  if (status === "error") return "red";
  if (status === "canceled") return "gray";
  if (status === "awaiting_review") return "orange";
  if (status && PROCESSING_STATUSES.has(status)) return "arcoblue";
  return "gray";
}

function progressHint(item: Partial<SessionSummary>) {
  if (item.error_message) return "处理失败，可点击重试或进入详情查看";
  if (item.is_processing) return "系统正在本地处理，请稍候";
  if (item.status === "awaiting_review") return "请确认咨询师 / 来访者角色";
  if (item.status === "complete") return "SOAP 与 Summary 已生成";
  return "等待处理";
}

function formatTime(value: number) {
  const total = Math.max(0, Math.round(Number(value) || 0));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function formatDateLabel(value?: string) {
  if (!value) return "未设置日期";
  const parts = String(value).split("-");
  if (parts.length !== 3) return value;
  const [year, month, day] = parts;
  return `${year}年${Number(month)}月${Number(day)}日`;
}

function normalizeSpeaker(value?: string): Speaker {
  return value === "Therapist" || value === "Client" ? value : "Unknown";
}

function summarizeWhisper(command: string) {
  if (!command) return "尚未配置";
  const modelMatch = command.match(/--model\s+([^\s]+)/);
  return `faster-whisper ${modelMatch ? modelMatch[1] : "本地模型"}`;
}

function summarizeLLM(command: string) {
  if (!command) return "尚未配置";
  const modelMatch = command.match(/--model\s+([^\s]+)/);
  return modelMatch ? `Ollama ${modelMatch[1]}` : "Ollama 本地模型";
}

function summarizeDiarization(command: string) {
  if (!command) return "未启用";
  if (command.includes("diarize_voice_features.py")) return "本地声音特征分组";
  return "已配置";
}

function StatusTag({ status, label }: { status?: string; label?: string }) {
  return <Tag color={statusColor(status)}>{label || status || "未知状态"}</Tag>;
}

function NoteList({ items }: { items?: string[] }) {
  const values = items && items.length ? items : ["逐字稿中未明确提及"];
  return (
    <ul className="note-list">
      {values.map((item, index) => (
        <li key={`${item}-${index}`}>{item}</li>
      ))}
    </ul>
  );
}

export default function App() {
  const [view, setView] = useState<ViewName>("records");
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSession, setCurrentSession] = useState<SessionDetail | null>(null);
  const [caseQuery, setCaseQuery] = useState("");
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [uploadVisible, setUploadVisible] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [savingSettings, setSavingSettings] = useState(false);
  const [transcriptDraft, setTranscriptDraft] = useState<TranscriptSegment[]>([]);
  const [uploadForm] = Form.useForm<UploadValues>();
  const [settingsForm] = Form.useForm<Settings>();
  const pollTimer = useRef<number | null>(null);

  const loadCases = useCallback(async () => {
    const data = await api<{ cases: CaseItem[] }>("/api/cases");
    setCases(data.cases || []);
  }, []);

  const loadSessions = useCallback(async () => {
    const query = new URLSearchParams();
    if (caseQuery.trim()) query.set("case", caseQuery.trim());
    setLoadingSessions(true);
    try {
      const data = await api<{ sessions: SessionSummary[] }>(`/api/sessions?${query.toString()}`);
      setSessions(data.sessions || []);
    } finally {
      setLoadingSessions(false);
    }
  }, [caseQuery]);

  const openSession = useCallback(async (caseId: string, sessionId: string) => {
    const data = await api<SessionDetail>(
      `/api/sessions/${encodeURIComponent(caseId)}/${encodeURIComponent(sessionId)}`
    );
    setCurrentSession(data);
    setTranscriptDraft(data.transcript || []);
    setView("session");
  }, []);

  const refreshCurrentSession = useCallback(async () => {
    if (!currentSession) return;
    const data = await api<SessionDetail>(
      `/api/sessions/${encodeURIComponent(currentSession.case_id)}/${encodeURIComponent(currentSession.session_id)}`
    );
    setCurrentSession(data);
    setTranscriptDraft(data.transcript || []);
  }, [currentSession]);

  const loadSettings = useCallback(async () => {
    try {
      const data = { ...DEFAULT_SETTINGS, ...(await api<Partial<Settings>>("/api/config")) };
      setSettings(data);
      settingsForm.setFieldsValue(data);
    } catch (error) {
      console.warn("Unable to load settings, showing defaults.", error);
      settingsForm.setFieldsValue(DEFAULT_SETTINGS);
    }
  }, [settingsForm]);

  useEffect(() => {
    loadCases().catch((error) => console.warn("Unable to load cases.", error));
    loadSettings();
  }, [loadCases, loadSettings]);

  useEffect(() => {
    loadSessions().catch((error) => {
      Message.error(`加载记录失败：${error.message}`);
    });
  }, [loadSessions]);

  useEffect(() => {
    if (pollTimer.current) window.clearTimeout(pollTimer.current);
    const shouldPoll =
      sessions.some((item) => item.is_processing) ||
      Boolean(currentSession && PROCESSING_STATUSES.has(currentSession.status || ""));
    if (shouldPoll) {
      pollTimer.current = window.setTimeout(async () => {
        await loadSessions();
        if (currentSession) await refreshCurrentSession();
      }, 2500);
    }
    return () => {
      if (pollTimer.current) window.clearTimeout(pollTimer.current);
    };
  }, [sessions, currentSession, loadSessions, refreshCurrentSession]);

  const retryTranscription = async (caseId: string, sessionId: string) => {
    await api(`/api/sessions/${caseId}/${sessionId}/transcribe`, { method: "POST" });
    Message.success("已重新加入转写队列");
    await loadSessions();
    if (currentSession?.session_id === sessionId) await refreshCurrentSession();
  };

  const rerunDiarization = async (caseId: string, sessionId: string) => {
    await api(`/api/sessions/${caseId}/${sessionId}/diarize`, { method: "POST" });
    Message.success("已开始重新自动整理");
    await loadSessions();
    await refreshCurrentSession();
  };

  const deleteSession = async (item: SessionSummary) => {
    await api(`/api/sessions/${item.case_id}/${item.session_id}`, { method: "DELETE" });
    Message.success("记录已删除");
    if (currentSession?.session_id === item.session_id) {
      setCurrentSession(null);
      setView("records");
    }
    await loadCases();
    await loadSessions();
  };

  const cancelSession = async (item: SessionSummary) => {
    await api(`/api/sessions/${item.case_id}/${item.session_id}/cancel`, { method: "POST" });
    Message.success("已取消当前处理任务");
    await loadSessions();
    if (currentSession?.session_id === item.session_id) await refreshCurrentSession();
  };

  const saveTranscript = async (data: SessionDetail) => {
    await api(`/api/sessions/${data.case_id}/${data.session_id}/transcript`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segments: transcriptDraft }),
    });
    Message.success("逐字稿修改已保存");
  };

  const submitReview = async (data: SessionDetail) => {
    const run = async () => {
      await api(`/api/sessions/${data.case_id}/${data.session_id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments: transcriptDraft }),
      });
      Message.success("已开始生成临床记录");
      await refreshCurrentSession();
      await loadSessions();
    };

    if (transcriptDraft.some((item) => !["Therapist", "Client"].includes(String(item.speaker)))) {
      Modal.confirm({
        title: "还有未确认角色",
        content: "仍有片段未确认成咨询师或来访者，仍然继续生成记录吗？",
        okText: "继续生成",
        cancelText: "返回检查",
        onOk: run,
      });
      return;
    }
    await run();
  };

  const submitUpload = async () => {
    const values = await uploadForm.validate();
    const file = values.audio?.[0]?.originFile;
    if (!file) {
      Message.warning("请选择录音文件");
      return;
    }
    const formData = new FormData();
    formData.append("case_id", values.case_id);
    formData.append("session_date", values.session_date || today());
    formData.append("note", values.note || "");
    formData.append("audio", file);
    setUploading(true);
    try {
      const session = await api<SessionSummary>("/api/sessions", { method: "POST", body: formData });
      Message.success("录音已上传，开始处理");
      setUploadVisible(false);
      uploadForm.resetFields();
      uploadForm.setFieldsValue({ session_date: today() });
      await loadCases();
      await loadSessions();
      await openSession(session.case_id, session.session_id);
    } finally {
      setUploading(false);
    }
  };

  const submitSettings = async () => {
    const values = await settingsForm.validate();
    setSavingSettings(true);
    try {
      const data = await api<Settings>("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      setSettings(data);
      settingsForm.setFieldsValue(data);
      Message.success("设置已保存");
    } finally {
      setSavingSettings(false);
    }
  };

  const columns = useMemo(
    () => [
      {
        title: "文件",
        dataIndex: "file_name",
        width: 360,
        render: (_: unknown, item: SessionSummary) => (
          <Space>
            <span className="file-mark">
              <IconFileAudio />
            </span>
            <div className="record-copy">
              <Text className="record-name">{item.file_name || "录音文件"}</Text>
              <Text type="secondary" className="record-id">
                {item.session_id}
              </Text>
            </div>
          </Space>
        ),
      },
      { title: "Case", dataIndex: "case_id", width: 130 },
      {
        title: "日期",
        dataIndex: "session_date",
        width: 150,
        render: (value: string) => value || "-",
      },
      {
        title: "状态",
        dataIndex: "status",
        width: 150,
        render: (_: unknown, item: SessionSummary) => <StatusTag status={item.status} label={item.progress_label} />,
      },
      {
        title: "操作",
        dataIndex: "actions",
        width: 300,
        fixed: "right" as const,
        render: (_: unknown, item: SessionSummary) => (
          <Space wrap>
            <Button type="text" size="small" onClick={() => openSession(item.case_id, item.session_id)}>
              查看详情
            </Button>
            {item.is_processing && (
              <Popconfirm
                title="取消只会停止当前本地处理，不会删除已经生成的文件。"
                okText="取消任务"
                cancelText="返回"
                onOk={() => cancelSession(item)}
              >
                <Button type="text" status="warning" size="small">
                  取消任务
                </Button>
              </Popconfirm>
            )}
            {item.next_action === "retry" && (
              <Button type="text" size="small" onClick={() => retryTranscription(item.case_id, item.session_id)}>
                重试
              </Button>
            )}
            <Popconfirm
              title={`确定删除“${item.file_name || "这条录音"}”吗？删除后不能恢复。`}
              okText="删除"
              cancelText="取消"
              onOk={() => deleteSession(item)}
            >
              <Button type="text" status="danger" size="small" icon={<IconDelete />}>
                删除
              </Button>
            </Popconfirm>
          </Space>
        ),
      },
    ],
    [openSession, currentSession, transcriptDraft]
  );

  return (
    <Layout className="app-shell">
      <Sider className="app-sider" width={224}>
        <div className="brand">
          <span className="brand-mark">M</span>
          <span>咨询记录</span>
        </div>
        <Menu selectedKeys={[view === "session" ? "records" : view]} onClickMenuItem={(key) => setView(key as ViewName)}>
          <Menu.Item key="records">
            <IconHome />
            记录
          </Menu.Item>
          <Menu.Item key="settings">
            <IconSettings />
            设置
          </Menu.Item>
        </Menu>
      </Sider>

      <Content className="app-content">
        {view === "records" && (
          <section className="page">
            <div className="page-header">
              <Title heading={4}>录音记录</Title>
              <Space>
                <Input.Search
                  allowClear
                  placeholder="搜索 Case ID"
                  value={caseQuery}
                  onChange={setCaseQuery}
                  className="case-search"
                />
                <Button
                  type="primary"
                  icon={<IconUpload />}
                  onClick={() => {
                    uploadForm.setFieldsValue({ session_date: today() });
                    setUploadVisible(true);
                  }}
                >
                  上传录音
                </Button>
              </Space>
            </div>

            <Table
              rowKey="session_id"
              loading={loadingSessions}
              columns={columns}
              data={sessions}
              pagination={false}
              scroll={{ x: 1090 }}
              noDataElement={<Empty description="还没有记录。点击右上角“上传录音”开始处理。" />}
            />
          </section>
        )}

        {view === "session" && currentSession && (
          <SessionDetailView
            data={currentSession}
            transcriptDraft={transcriptDraft}
            setTranscriptDraft={setTranscriptDraft}
            onBack={async () => {
              setCurrentSession(null);
              setView("records");
              await loadSessions();
            }}
            onRefresh={refreshCurrentSession}
            onRetry={() => retryTranscription(currentSession.case_id, currentSession.session_id)}
            onRerunDiarization={() => rerunDiarization(currentSession.case_id, currentSession.session_id)}
            onRegenerate={() => submitReview(currentSession)}
            onSaveTranscript={async () => {
              await saveTranscript(currentSession);
              await refreshCurrentSession();
              await loadSessions();
            }}
            onSubmitReview={() => submitReview(currentSession)}
          />
        )}

        {view === "settings" && (
          <section className="page settings-page">
            <Title heading={4}>设置</Title>
            <Form form={settingsForm} layout="vertical" initialValues={settings} onSubmit={submitSettings}>
              <div className="settings-block">
                <Title heading={6}>本地资料库</Title>
                <Row gutter={16}>
                  <Col span={12}>
                    <FormItem label="数据保存位置" field="data_root" rules={[{ required: true, message: "请输入数据保存位置" }]}>
                      <Input placeholder="data" />
                    </FormItem>
                  </Col>
                  <Col span={12}>
                    <FormItem label="Prompt 文件" field="clinical_prompt_path" rules={[{ required: true, message: "请输入 Prompt 文件" }]}>
                      <Input placeholder="prompts/clinical_note_prompt.md" />
                    </FormItem>
                  </Col>
                </Row>
              </div>

              <div className="settings-block">
                <Title heading={6}>本地模型</Title>
                <Descriptions
                  column={1}
                  data={[
                    { label: "转写模型", value: summarizeWhisper(settingsForm.getFieldValue("whisper_command") || settings.whisper_command) },
                    {
                      label: "说话人区分",
                      value: summarizeDiarization(settingsForm.getFieldValue("diarization_command") || settings.diarization_command),
                    },
                    { label: "本地大模型", value: summarizeLLM(settingsForm.getFieldValue("llm_command") || settings.llm_command) },
                  ]}
                />
              </div>

              <Collapse className="settings-block" defaultActiveKey={[]}>
                <Collapse.Item header="高级命令模板" name="commands">
                  <FormItem label="Whisper 命令" field="whisper_command">
                    <TextArea autoSize={{ minRows: 3, maxRows: 6 }} />
                  </FormItem>
                  <FormItem label="说话人区分命令" field="diarization_command">
                    <TextArea autoSize={{ minRows: 3, maxRows: 6 }} />
                  </FormItem>
                  <FormItem label="LLM 命令" field="llm_command">
                    <TextArea autoSize={{ minRows: 3, maxRows: 6 }} />
                  </FormItem>
                </Collapse.Item>
              </Collapse>

              <div className="settings-actions">
                <Button type="primary" htmlType="submit" loading={savingSettings}>
                  保存设置
                </Button>
              </div>
            </Form>
          </section>
        )}
      </Content>

      <Modal
        title="上传录音"
        visible={uploadVisible}
        onCancel={() => setUploadVisible(false)}
        onOk={submitUpload}
        confirmLoading={uploading}
        okText="上传并处理"
        cancelText="取消"
        unmountOnExit
      >
        <Form form={uploadForm} layout="vertical" initialValues={{ session_date: today() }}>
          <FormItem label="Case ID" field="case_id" rules={[{ required: true, message: "请输入 Case ID" }]}>
            <Input placeholder="例如 C-001" list="caseOptions" />
          </FormItem>
          <datalist id="caseOptions">
            {cases.map((item) => (
              <option key={item.case_id} value={item.case_id} />
            ))}
          </datalist>
          <FormItem label="会谈日期" field="session_date" rules={[{ required: true, message: "请选择会谈日期" }]}>
            <DatePicker className="full-width" onChange={(dateString) => uploadForm.setFieldValue("session_date", dateString)} />
          </FormItem>
          <FormItem label="Case 备注" field="note">
            <TextArea autoSize={{ minRows: 2, maxRows: 4 }} placeholder="可选" />
          </FormItem>
          <FormItem label="录音文件" field="audio" rules={[{ required: true, message: "请选择录音文件" }]}>
            <Upload
              drag
              limit={1}
              accept=".mp3,.wav,.m4a,audio/*"
              autoUpload={false}
              tip="支持 mp3 / wav / m4a。上传后会自动开始转写。"
            />
          </FormItem>
        </Form>
      </Modal>
    </Layout>
  );
}

function SessionDetailView({
  data,
  transcriptDraft,
  setTranscriptDraft,
  onBack,
  onRefresh,
  onRetry,
  onRerunDiarization,
  onRegenerate,
  onSaveTranscript,
  onSubmitReview,
}: {
  data: SessionDetail;
  transcriptDraft: TranscriptSegment[];
  setTranscriptDraft: (segments: TranscriptSegment[]) => void;
  onBack: () => void;
  onRefresh: () => void;
  onRetry: () => void;
  onRerunDiarization: () => void;
  onRegenerate: () => void;
  onSaveTranscript: () => void;
  onSubmitReview: () => void;
}) {
  const disabled = data.status === "generating_note";

  const setSegment = (index: number, patch: Partial<TranscriptSegment>) => {
    setTranscriptDraft(transcriptDraft.map((item, current) => (current === index ? { ...item, ...patch } : item)));
  };

  return (
    <section className="page session-page">
      <div className="detail-header">
        <Button onClick={onBack}>返回记录</Button>
        <div className="detail-title">
          <Title heading={4}>{`${data.case_id || "-"}，${formatDateLabel(data.session_date)}`}</Title>
          <StatusTag status={data.status} label={data.progress_label} />
        </div>
        <Space wrap className="detail-actions">
          {data.status === "error" && <Button onClick={onRetry}>重新转写</Button>}
          {Boolean(data.transcript?.length && data.diarization_configured && !data.is_processing) && (
            <Button onClick={onRerunDiarization}>重新自动整理</Button>
          )}
          {data.status === "complete" && Boolean(data.transcript?.length) && <Button onClick={onRegenerate}>重新生成记录</Button>}
          <Button icon={<IconRefresh />} onClick={onRefresh}>
            刷新
          </Button>
        </Space>
      </div>

      <div className="detail-grid">
        <aside className="summary-pane">
          <Tabs defaultActiveTab="summary">
            <Tabs.TabPane key="summary" title="Summary">
              <ClinicalSummary note={data.clinical_note || null} />
            </Tabs.TabPane>
            <Tabs.TabPane key="soap" title="SOAP">
              <SoapNote note={data.clinical_note || null} />
            </Tabs.TabPane>
            <Tabs.TabPane key="paths" title="文件">
              <Descriptions
                column={1}
                data={[
                  { label: "音频", value: data.audio_path || "-" },
                  { label: "逐字稿", value: data.transcript_path || "-" },
                  { label: "JSON", value: data.note_json_path || "-" },
                  { label: "Markdown", value: data.markdown_path || "-" },
                ]}
              />
            </Tabs.TabPane>
          </Tabs>
        </aside>

        <section className="transcript-pane">
          <div className="pane-head">
            <Title heading={5}>文字记录</Title>
            <Text type="secondary">
              {transcriptDraft.length
                ? data.diarization_configured
                  ? "系统已自动分段并标注角色，可直接修改错误的角色。"
                  : "系统已生成逐字稿，可直接修改角色。"
                : `逐字稿生成后会显示在这里。${progressHint(data)}`}
            </Text>
          </div>
          <Spin loading={Boolean(data.is_processing && !transcriptDraft.length)} className="transcript-spin">
            <div className="transcript-list">
              {!transcriptDraft.length ? (
                <Empty description="还没有逐字稿。" />
              ) : (
                transcriptDraft.map((item, index) => (
                  <div className="segment" key={`${item.start}-${item.end}-${index}`}>
                    <Text type="secondary" className="segment-time">
                      {formatTime(item.start)} - {formatTime(item.end)}
                    </Text>
                    <Select
                      value={normalizeSpeaker(String(item.speaker || "Unknown"))}
                      disabled={disabled}
                      onChange={(speaker) => setSegment(index, { speaker })}
                    >
                      <Select.Option value="Therapist">咨询师</Select.Option>
                      <Select.Option value="Client">来访者</Select.Option>
                      <Select.Option value="Unknown">未确认</Select.Option>
                    </Select>
                    <TextArea
                      value={item.text || ""}
                      disabled={disabled}
                      autoSize={{ minRows: 2, maxRows: 8 }}
                      onChange={(text) => setSegment(index, { text })}
                    />
                  </div>
                ))
              )}
            </div>
          </Spin>
          <div className="review-bar">
            <Text type="secondary">
              {data.status === "awaiting_review"
                ? "修改后可以先保存，也可以直接生成记录。"
                : data.status === "complete"
                  ? "记录已生成。如需修改，可保存后重新生成记录。"
                  : progressHint(data)}
            </Text>
            <Space>
              {(data.status === "awaiting_review" || data.status === "complete") && (
                <Button icon={<IconEdit />} onClick={onSaveTranscript}>
                  保存修改
                </Button>
              )}
              {data.status === "awaiting_review" && (
                <Button type="primary" onClick={onSubmitReview}>
                  生成记录
                </Button>
              )}
            </Space>
          </div>
        </section>
      </div>
    </section>
  );
}

function ClinicalSummary({ note }: { note: ClinicalNote | null }) {
  if (!note) return <Empty description="确认角色后，系统会生成 Session Summary。" />;
  return (
    <div className="note-stack">
      {["本次主题", "核心问题", "情绪变化", "咨询师主要回应方式", "会谈结构", "关键转折点"].map((key) => (
        <section className="note-section" key={key}>
          <Title heading={6}>{key}</Title>
          <NoteList items={note.session_summary?.[key]} />
        </section>
      ))}
    </div>
  );
}

function SoapNote({ note }: { note: ClinicalNote | null }) {
  if (!note) return <Empty description="确认角色后，系统会生成 SOAP 记录。" />;
  return (
    <div className="note-stack">
      {["S", "O", "A", "P"].map((key) => (
        <section className="note-section" key={key}>
          <Title heading={6}>{key}</Title>
          <NoteList items={note.soap?.[key]} />
        </section>
      ))}
    </div>
  );
}
