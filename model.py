import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34
from transformers import BertTokenizer, BertModel
import clip


class SiameseBaselineModel(torch.nn.Module):
    def __init__(
        self,
        weight_lang_loss=0.3,
        t=0.07,
        use_clip=True,
        share_weights=False,
        use_rnn=True,
        use_crop=True,
        is_triplet_loss=False,
    ):

        super().__init__()
        self.weight_lang_loss = weight_lang_loss
        self.t = t
        self.use_clip = use_clip
        self.share_weights = share_weights
        self.use_rnn = use_rnn
        self.use_crop = use_crop
        self.is_triplet_loss = is_triplet_loss

        if not self.use_clip:
            self.bg_model = resnet50(pretrained=True, num_classes=1000)
            self.bg_model.fc = nn.Linear(2048, 512)
            self.bg_model.half().cuda()

            if self.share_weights:
                self.crop_model = self.bg_model
            else:
                self.crop_model = resnet34(pretrained=True, num_classes=1000)
                self.crop_model.fc = nn.Linear(512, 512)
                self.crop_model.half().cuda()

            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased").half()
            self.lang_fc = torch.nn.Linear(768, 256)
            self.lang_fc.half().cuda()

        else:
            self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
            self.bg_model = self.clip_model.encode_image
            if self.share_weights:
                self.crop_model = self.bg_model
            else:
                self.clip_model_crop, _ = clip.load("ViT-B/32", device="cuda")
                self.crop_model = self.clip_model_crop.encode_image
            self.lang_fc = torch.nn.Linear(512, 256).cuda().half()

        if self.use_crop and self.use_rnn:
            self.linear_project = nn.Linear(512 * 2 + 128, 256)
        elif self.use_crop:
            self.linear_project = nn.Linear(512 * 2, 256)
        elif self.use_rnn:
            self.linear_project = nn.Linear(512 + 128, 256)
        else:
            self.linear_project = nn.Linear(512, 256)

        self.linear_project.half().cuda()

        self.loss_fn = (
            torch.nn.TripletMarginLoss() if self.is_triplet_loss else self.nce_loss
        )

        if self.use_rnn:
            self.rnn = nn.LSTM(
                input_size=2,
                hidden_size=64,
                num_layers=2,
                bias=True,
                batch_first=True,
                dropout=0.1,
                bidirectional=True,
            )
            self.rnn.half().cuda()

    def forward(self, track):
        bs = track["frame_img"].shape[0]
        nl = deepcopy(track["nl"])
        for i in range(bs):
            nl += [item[i] for item in track["neg_nl"]]
        # nl = deepcopy(track["nl"])
        # nl += track['neg_nl']

        if not self.use_clip:
            tokens = self.bert_tokenizer.batch_encode_plus(
                nl, padding="longest", return_tensors="pt"
            )

            outputs = self.bert_model(
                tokens["input_ids"].cuda(),
                attention_mask=tokens["attention_mask"].cuda(),
            )
            lang_embeds = torch.mean(outputs.last_hidden_state.half(), dim=1)
            lang_embeds = self.lang_fc(lang_embeds)

        else:
            tokens = clip.tokenize(nl).cuda()
            outputs = self.clip_model.encode_text(tokens)
            lang_embeds = self.lang_fc(outputs)

        frame_imgs = torch.cat(
            [
                track["frame_img"],
                track["neg_frames"].view(-1, *track["neg_frames"].shape[-3:]),
            ],
            dim=0,
        )
        if not self.use_clip:
            frame_imgs = frame_imgs.half()
        frame_embeds = self.bg_model(frame_imgs.cuda())

        if self.use_crop:
            crops = torch.cat(
                [
                    track["frame_crop_img"],
                    track["neg_crops"].view(-1, *track["neg_crops"].shape[-3:]),
                ],
                dim=0,
            )
            if not self.use_clip:
                crops = crops.half()
            crops_embeds = self.crop_model(crops.cuda())

        if self.use_rnn:
            boxes = torch.cat(
                [
                    track["boxes_points"],
                    track["neg_boxes_points"].view(
                        -1, *track["neg_boxes_points"].shape[-2:]
                    ),
                ],
                dim=0,
            )
            boxes_embeds = self.rnn(boxes.half().cuda())[0][:, 0, :]

        if self.use_crop and self.use_rnn:
            concat = torch.cat([frame_embeds, crops_embeds, boxes_embeds], dim=-1)
        elif self.use_crop:
            concat = torch.cat([frame_embeds, crops_embeds], dim=-1)
        elif self.use_rnn:
            concat = torch.cat([frame_embeds, boxes_embeds], dim=-1)
        else:
            concat = frame_embeds

        visual_embeds = self.linear_project(F.relu(concat))

        if self.is_triplet_loss:
            lang_embeds = F.normalize(lang_embeds, dim=-1)
            visual_embeds = F.normalize(visual_embeds, dim=-1)
            neg_lang_embeds = lang_embeds[bs: 2 * bs]
            neg_visual_embeds = visual_embeds[bs: 2 * bs]
        else:
            neg_lang_embeds = lang_embeds[bs:].view(bs, -1, 256)
            neg_visual_embeds = visual_embeds[bs:].view(bs, -1, 256)

        query_lang_embeds = lang_embeds[:bs]
        query_visual_embeds = visual_embeds[:bs]

        return {
            "ql": query_lang_embeds,
            "qv": query_visual_embeds,
            "nl": neg_lang_embeds,
            "nv": neg_visual_embeds,
        }

    def nce_loss(self, q, po, ne):
        N = q.shape[0]
        C = q.shape[1]
        M = ne.shape[1]
        # If mat1 is a b×n×m tensor, mat2 is a b×m×p tensor,
        # then output will be a b×n×p tensor.
        q_norm = torch.norm(q, dim=1)
        pos_pair_norm = q_norm * torch.norm(po, dim=1).view(
            N,
        )
        neg_pair_norm = q_norm.view(N, 1) * torch.norm(ne, dim=-1).view(N, M)

        pos = torch.exp(
            torch.div(
                torch.bmm(q.view(N, 1, C), po.view(N, C, 1)).view(
                    N,
                )
                / pos_pair_norm,
                self.t,
            )
        )
        neg = torch.sum(
            torch.exp(
                torch.div(
                    torch.bmm(q.view(N, 1, C), ne.permute(0, 2, 1)).view(N, M)
                    / neg_pair_norm,
                    self.t,
                )
            ),
            dim=1,
        )
        return torch.mean(-torch.log(pos / (neg + pos) + 1e-5))

    def compute_vi_embed(self, track):
        with torch.no_grad():
            # print(track)
            frame_embeds = (
                self.bg_model(track["frame_img"].cuda())
                if self.use_clip
                else self.bg_model(track["frame_img"].half().cuda())
            )
            if self.use_crop:
                crops_embeds = (
                    self.crop_model(track["frame_crop_img"].cuda())
                    if self.use_clip
                    else self.crop_model(track["frame_crop_img"].half().cuda())
                )
            if self.use_rnn:
                box_embeds = self.rnn(track["boxes_points"].half().cuda())[0][:, 0, :]

            if self.use_crop and self.use_rnn:
                concat = torch.cat([frame_embeds, crops_embeds, box_embeds], dim=-1)
            elif self.use_crop:
                concat = torch.cat([frame_embeds, crops_embeds], dim=-1)
            elif self.use_rnn:
                concat = torch.cat([frame_embeds, box_embeds], dim=-1)
            else:
                concat = frame_embeds

            visual_embeds = self.linear_project(F.relu(concat))
            if self.is_triplet_loss:
                visual_embeds = F.normalize(visual_embeds)

            return visual_embeds

    def compute_loss(self, track):
        embeds = self.forward(track)
        nce_visual_loss = self.loss_fn(embeds["qv"], embeds["ql"], embeds["nl"])
        nce_lang_loss = self.loss_fn(embeds["ql"], embeds["qv"], embeds["nv"])
        return (
            1 - self.weight_lang_loss
        ) * nce_visual_loss + self.weight_lang_loss * nce_lang_loss

    def compute_lang_embed(self, nls):
        with torch.no_grad():
            if self.use_clip:
                tokens = clip.tokenize(nls).cuda()
                outputs = self.clip_model.encode_text(tokens)
                lang_embeds = self.lang_fc(outputs)

            else:
                tokens = self.bert_tokenizer.batch_encode_plus(
                    nls, padding="longest", return_tensors="pt"
                )
                outputs = self.bert_model(
                    tokens["input_ids"].cuda(),
                    attention_mask=tokens["attention_mask"].cuda(),
                )
                lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
                lang_embeds = self.lang_fc(lang_embeds)
            if self.is_triplet_loss:
                lang_embeds = F.normalize(lang_embeds)

        return lang_embeds
