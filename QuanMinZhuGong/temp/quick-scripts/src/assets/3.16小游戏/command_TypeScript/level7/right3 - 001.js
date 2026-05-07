"use strict";
cc._RF.push(module, 'f0e01u1aPhDp4Dw+P9f7QGg', 'right3 - 001');
// 3.16小游戏/command_TypeScript/level7/right3 - 001.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___003_1 = require("./global - 003");
var enemy_3 = /** @class */ (function (_super) {
    __extends(enemy_3, _super);
    function enemy_3() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_3.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_3.prototype.start = function () {
        this.schedule(function () {
            global___003_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_3.prototype.onCollisionEnter = function (other, self) {
        if (global___003_1.default.minion_attack == true) {
            global___003_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_3.prototype.onCollisionExit = function (other) {
        global___003_1.default.minion_attack = false;
        if (global___003_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___003_1.default.main_hp, 19);
        }
    };
    enemy_3.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼2");
        if (this.count == 0) {
            if (node1.active == false) {
                this.node.setPosition(this.node.position.x, node1.position.y);
                var node2 = cc.find("Canvas/bj/b/a");
                node2.setContentSize(400, 26);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (global___003_1.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_3 = __decorate([
        ccclass
    ], enemy_3);
    return enemy_3;
}(cc.Component));
exports.default = enemy_3;

cc._RF.pop();